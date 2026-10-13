package main

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"
)

const (
	defaultModel = "openai/gpt-oss-20b"
	maxHistory   = 20 // keep only the last N messages per user
	reqTimeout   = 60 * time.Second
)

type ChatRequest struct {
	Message string `json:"message"`
}

type ChatResponse struct {
	Reply string `json:"reply"`
}

var (
	chatHistory = make(map[string][]openai.ChatCompletionMessage)
	mutex       sync.Mutex
)

// addUserMessage stores the user's message and returns a safe copy of the
// (trimmed) history to send to the model.
func addUserMessage(userID, text string) []openai.ChatCompletionMessage {
	mutex.Lock()
	defer mutex.Unlock()

	h := append(chatHistory[userID], openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: text,
	})
	if len(h) > maxHistory {
		h = h[len(h)-maxHistory:]
		for len(h) > 1 && h[0].Role != openai.ChatMessageRoleUser {
			h = h[1:]
		}
	}
	chatHistory[userID] = h
	return append([]openai.ChatCompletionMessage(nil), h...)
}

func addAssistantMessage(userID, text string) {
	mutex.Lock()
	defer mutex.Unlock()
	chatHistory[userID] = append(chatHistory[userID], openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: text,
	})
}

// removeLastUserMessage rolls back a user message when the model call fails,
// so history never contains a user message without a reply.
func removeLastUserMessage(userID string) {
	mutex.Lock()
	defer mutex.Unlock()
	h := chatHistory[userID]
	if n := len(h); n > 0 && h[n-1].Role == openai.ChatMessageRoleUser {
		chatHistory[userID] = h[:n-1]
	}
}

// ---------- Authentication (users and sessions stored in Upstash Redis) ----------

var (
	redisURL, redisToken string
	httpClient           = &http.Client{Timeout: 10 * time.Second}
	usernameRe           = regexp.MustCompile(`^[a-z0-9_]{3,20}$`)
)

const (
	pbkdf2Iterations = 120000
	sessionSeconds   = 604800 // 7 days
)

type Credentials struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

// redis sends one Redis command through Upstash's REST API.
func redis(args ...any) (any, error) {
	body, err := json.Marshal(args)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest(http.MethodPost, redisURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+redisToken)
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var out struct {
		Result any    `json:"result"`
		Error  string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if out.Error != "" {
		return nil, errors.New(out.Error)
	}
	return out.Result, nil
}

// pbkdf2Sha256 derives a 32-byte key (one HMAC-SHA256 block) from a password.
func pbkdf2Sha256(password, salt []byte, iter int) []byte {
	prf := hmac.New(sha256.New, password)
	prf.Write(salt)
	prf.Write([]byte{0, 0, 0, 1})
	u := prf.Sum(nil)
	key := append([]byte(nil), u...)
	for i := 1; i < iter; i++ {
		prf.Reset()
		prf.Write(u)
		u = prf.Sum(nil)
		for j := range key {
			key[j] ^= u[j]
		}
	}
	return key
}

func hashPassword(password string) string {
	salt := make([]byte, 16)
	rand.Read(salt)
	return hex.EncodeToString(salt) + "$" + hex.EncodeToString(pbkdf2Sha256([]byte(password), salt, pbkdf2Iterations))
}

func checkPassword(password, stored string) bool {
	parts := strings.SplitN(stored, "$", 2)
	if len(parts) != 2 {
		return false
	}
	salt, err := hex.DecodeString(parts[0])
	if err != nil {
		return false
	}
	want, err := hex.DecodeString(parts[1])
	if err != nil {
		return false
	}
	got := pbkdf2Sha256([]byte(password), salt, pbkdf2Iterations)
	return subtle.ConstantTimeCompare(got, want) == 1
}

func newToken() string {
	b := make([]byte, 32)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func bearerToken(r *http.Request) string {
	return strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
}

// authUser returns the username that owns the session token, if it is valid.
func authUser(r *http.Request) (string, bool) {
	token := bearerToken(r)
	if token == "" {
		return "", false
	}
	res, err := redis("GET", "sess:"+token)
	if err != nil {
		log.Println("Redis error:", err)
		return "", false
	}
	user, _ := res.(string)
	return user, user != ""
}

func readCredentials(w http.ResponseWriter, r *http.Request) (Credentials, bool) {
	var c Credentials
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
		return c, false
	}
	r.Body = http.MaxBytesReader(w, r.Body, 1<<16)
	if err := json.NewDecoder(r.Body).Decode(&c); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return c, false
	}
	c.Username = strings.ToLower(strings.TrimSpace(c.Username))
	return c, true
}

func registerHandler(w http.ResponseWriter, r *http.Request) {
	c, ok := readCredentials(w, r)
	if !ok {
		return
	}
	if !usernameRe.MatchString(c.Username) {
		http.Error(w, "Username must be 3-20 characters: letters, numbers or underscore", http.StatusBadRequest)
		return
	}
	if len(c.Password) < 6 {
		http.Error(w, "Password must be at least 6 characters", http.StatusBadRequest)
		return
	}
	// NX = only set if the key does not exist yet, so usernames stay unique.
	res, err := redis("SET", "user:"+c.Username, hashPassword(c.Password), "NX")
	if err != nil {
		log.Println("Register error:", err)
		http.Error(w, "Database error", http.StatusInternalServerError)
		return
	}
	if res == nil {
		http.Error(w, "Username already taken", http.StatusConflict)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]string{"message": "registered"})
}

func loginHandler(w http.ResponseWriter, r *http.Request) {
	c, ok := readCredentials(w, r)
	if !ok {
		return
	}
	res, err := redis("GET", "user:"+c.Username)
	if err != nil {
		log.Println("Login error:", err)
		http.Error(w, "Database error", http.StatusInternalServerError)
		return
	}
	stored, _ := res.(string)
	if stored == "" || !checkPassword(c.Password, stored) {
		http.Error(w, "Invalid username or password", http.StatusUnauthorized)
		return
	}
	token := newToken()
	if _, err := redis("SET", "sess:"+token, c.Username, "EX", sessionSeconds); err != nil {
		log.Println("Session error:", err)
		http.Error(w, "Database error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"token": token, "username": c.Username})
}

func logoutHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
		return
	}
	if token := bearerToken(r); token != "" {
		if _, err := redis("DEL", "sess:"+token); err != nil {
			log.Println("Logout error:", err)
		}
	}
	w.WriteHeader(http.StatusNoContent)
}

// decodeChatRequest checks method, login token and body. The user comes from the
// verified session, never from the request body, so it cannot be spoofed.
func decodeChatRequest(w http.ResponseWriter, r *http.Request) (string, ChatRequest, bool) {
	var req ChatRequest
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
		return "", req, false
	}
	userID, ok := authUser(r)
	if !ok {
		http.Error(w, "Please log in", http.StatusUnauthorized)
		return "", req, false
	}
	r.Body = http.MaxBytesReader(w, r.Body, 1<<16)
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return "", req, false
	}
	if strings.TrimSpace(req.Message) == "" {
		http.Error(w, "message required", http.StatusBadRequest)
		return "", req, false
	}
	return userID, req, true
}

func sseSend(w http.ResponseWriter, f http.Flusher, event string, payload any) {
	data, _ := json.Marshal(payload)
	if event != "" {
		fmt.Fprintf(w, "event: %s\n", event)
	}
	fmt.Fprintf(w, "data: %s\n\n", data)
	f.Flush()
}

func main() {
	_ = godotenv.Load()

	redisURL = os.Getenv("UPSTASH_REDIS_REST_URL")
	redisToken = os.Getenv("UPSTASH_REDIS_REST_TOKEN")
	if redisURL == "" || redisToken == "" {
		log.Fatal("UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN must be set")
	}

	apiKey := os.Getenv("GROQ_API_KEY")
	if apiKey == "" {
		log.Fatal("GROQ_API_KEY not set")
	}

	// Model name comes from the environment so a deprecation needs no code change.
	model := os.Getenv("GROQ_MODEL")
	if model == "" {
		model = defaultModel
	}

	config := openai.DefaultConfig(apiKey)
	config.BaseURL = "https://api.groq.com/openai/v1"
	client := openai.NewClientWithConfig(config)

	mux := http.NewServeMux()

	// Auth API
	mux.HandleFunc("/register", registerHandler)
	mux.HandleFunc("/login", loginHandler)
	mux.HandleFunc("/logout", logoutHandler)

	// Serve frontend
	mux.Handle("/", http.FileServer(http.Dir("./static")))

	// Standard endpoint: returns the full reply as JSON (frontend keeps working).
	mux.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
		userID, req, ok := decodeChatRequest(w, r)
		if !ok {
			return
		}

		messages := addUserMessage(userID, req.Message)

		ctx, cancel := context.WithTimeout(r.Context(), reqTimeout)
		defer cancel()

		resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
			Model:    model,
			Messages: messages,
		})
		if err != nil {
			log.Println("Groq error:", err)
			removeLastUserMessage(userID)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if len(resp.Choices) == 0 {
			log.Println("Groq returned no choices")
			removeLastUserMessage(userID)
			http.Error(w, "Empty response from model", http.StatusInternalServerError)
			return
		}

		reply := resp.Choices[0].Message.Content
		addAssistantMessage(userID, reply)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{Reply: reply})
	})

	// Streaming endpoint: sends the reply token by token using Server-Sent Events.
	mux.HandleFunc("/chat/stream", func(w http.ResponseWriter, r *http.Request) {
		userID, req, ok := decodeChatRequest(w, r)
		if !ok {
			return
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		messages := addUserMessage(userID, req.Message)

		// r.Context() is cancelled if the browser disconnects, which stops the Groq call.
		ctx, cancel := context.WithTimeout(r.Context(), reqTimeout)
		defer cancel()

		stream, err := client.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
			Model:    model,
			Messages: messages,
		})
		if err != nil {
			log.Println("Groq stream error:", err)
			removeLastUserMessage(userID)
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer stream.Close()

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no") // stop proxies from buffering the stream

		var full strings.Builder
		for {
			chunk, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				log.Println("Groq stream recv error:", err)
				removeLastUserMessage(userID)
				sseSend(w, flusher, "error", map[string]string{"error": err.Error()})
				return
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			token := chunk.Choices[0].Delta.Content
			if token == "" {
				continue
			}
			full.WriteString(token)
			sseSend(w, flusher, "", map[string]string{"token": token})
		}

		addAssistantMessage(userID, full.String())
		sseSend(w, flusher, "done", map[string]bool{"done": true})
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// No WriteTimeout on purpose: it would cut long streaming responses.
	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Println("🚀 Server running on port", port)
	log.Fatal(srv.ListenAndServe())
}
