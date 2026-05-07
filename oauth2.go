/*
Agent Runtime — Fantasy (Go)

OAuth2 client_credentials token transport.
Replaces the token-injector sidecar by fetching and caching tokens inline.
*/
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

// oauth2Transport is an http.RoundTripper that transparently injects
// a Bearer token obtained via OAuth2 client_credentials grant.
type oauth2Transport struct {
	tokenURL     string
	clientID     string
	clientSecret string
	scopes       []string
	base         http.RoundTripper

	mu      sync.Mutex
	token   string
	expiry  time.Time
}

func (t *oauth2Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	token, err := t.getToken()
	if err != nil {
		return nil, fmt.Errorf("oauth2: token fetch failed: %w", err)
	}
	req = req.Clone(req.Context())
	req.Header.Set("Authorization", "Bearer "+token)
	return t.base.RoundTrip(req)
}

func (t *oauth2Transport) getToken() (string, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Return cached token if still valid (with 30s margin).
	if t.token != "" && time.Now().Add(30*time.Second).Before(t.expiry) {
		return t.token, nil
	}

	// Fetch new token.
	data := url.Values{
		"grant_type":    {"client_credentials"},
		"client_id":     {t.clientID},
		"client_secret": {t.clientSecret},
	}
	if len(t.scopes) > 0 {
		data.Set("scope", strings.Join(t.scopes, " "))
	}

	resp, err := http.PostForm(t.tokenURL, data)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("token endpoint returned %d: %s", resp.StatusCode, string(body))
	}

	var tok struct {
		AccessToken string `json:"access_token"`
		ExpiresIn   int    `json:"expires_in"`
	}
	if err := json.Unmarshal(body, &tok); err != nil {
		return "", fmt.Errorf("failed to parse token response: %w", err)
	}
	if tok.AccessToken == "" {
		return "", fmt.Errorf("empty access_token in response")
	}

	t.token = tok.AccessToken
	if tok.ExpiresIn > 0 {
		t.expiry = time.Now().Add(time.Duration(tok.ExpiresIn) * time.Second)
	} else {
		t.expiry = time.Now().Add(5 * time.Minute) // conservative default
	}
	return t.token, nil
}

// newOAuth2HTTPClient creates an *http.Client with OAuth2 token injection.
// ClientID and ClientSecret are resolved from the environment variables
// specified in the OAuth2Entry (populated by the operator from K8s Secrets).
func newOAuth2HTTPClient(cfg *OAuth2Entry) *http.Client {
	clientID := os.Getenv(cfg.ClientIDEnv)
	clientSecret := os.Getenv(cfg.ClientSecretEnv)
	return &http.Client{
		Transport: &oauth2Transport{
			tokenURL:     cfg.TokenURL,
			clientID:     clientID,
			clientSecret: clientSecret,
			scopes:       cfg.Scopes,
			base:         http.DefaultTransport,
		},
		Timeout: 5 * time.Minute,
	}
}
