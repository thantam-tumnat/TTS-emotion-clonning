package eventlog

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// Record is one append-only JSONL entry. Keep it small: job payloads belong in
// the job API, while this file is the durable timeline used by the dashboard.
type Record struct {
	Time    string                 `json:"time"`
	Service string                 `json:"service"`
	Level   string                 `json:"level"`
	Event   string                 `json:"event"`
	JobID   string                 `json:"job_id,omitempty"`
	Message string                 `json:"message,omitempty"`
	Fields  map[string]interface{} `json:"fields,omitempty"`
}

type Logger struct {
	mu      sync.Mutex
	dir     string
	service string
	date    string
	file    *os.File
}

func New(dir, service string) (*Logger, error) {
	if dir == "" {
		dir = "logs"
	}
	if service == "" {
		service = "service"
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &Logger{dir: dir, service: service}, nil
}

func (l *Logger) Directory() string { return l.dir }

func (l *Logger) Write(level, event, jobID, message string, fields map[string]interface{}, syncDisk bool) error {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	now := time.Now()
	date := now.Format("2006-01-02")
	if l.file == nil || l.date != date {
		if l.file != nil {
			_ = l.file.Close()
		}
		name := filepath.Join(l.dir, fmt.Sprintf("%s-%s.jsonl", safePart(l.service), date))
		f, err := os.OpenFile(name, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		l.file = f
		l.date = date
	}

	record := Record{
		Time: now.Format(time.RFC3339Nano), Service: l.service,
		Level: strings.ToUpper(level), Event: event, JobID: jobID,
		Message: message, Fields: fields,
	}
	b, err := json.Marshal(record)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if _, err = l.file.Write(b); err != nil {
		return err
	}
	if syncDisk {
		return l.file.Sync()
	}
	return nil
}

func (l *Logger) Close() error {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.file == nil {
		return nil
	}
	err := l.file.Sync()
	closeErr := l.file.Close()
	l.file = nil
	return errors.Join(err, closeErr)
}

func safePart(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "service"
	}
	return regexp.MustCompile(`[^a-zA-Z0-9._-]+`).ReplaceAllString(value, "_")
}

type QueryOptions struct {
	Date    string
	Service string
	Level   string
	JobID   string
	Event   string
	Query   string
	Limit   int
}

func Query(dir string, opts QueryOptions) ([]Record, error) {
	if dir == "" {
		dir = "logs"
	}
	if opts.Date == "" {
		opts.Date = time.Now().Format("2006-01-02")
	}
	if !regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`).MatchString(opts.Date) {
		return nil, fmt.Errorf("invalid date")
	}
	if opts.Limit <= 0 || opts.Limit > 1000 {
		opts.Limit = 200
	}
	pattern := "*-" + opts.Date + ".jsonl"
	if opts.Service != "" {
		pattern = safePart(opts.Service) + "-" + opts.Date + ".jsonl"
	}
	files, err := filepath.Glob(filepath.Join(dir, pattern))
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	var matches []Record
	for _, name := range files {
		f, err := os.Open(name)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return nil, err
		}
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 64*1024), 2*1024*1024)
		for scanner.Scan() {
			var record Record
			if json.Unmarshal(scanner.Bytes(), &record) != nil {
				continue // tolerate a partially written final line after a hard stop
			}
			if !matchesRecord(record, opts) {
				continue
			}
			matches = append(matches, record)
			if len(matches) > opts.Limit {
				matches = matches[1:]
			}
		}
		closeErr := f.Close()
		if err := scanner.Err(); err != nil {
			return nil, err
		}
		if closeErr != nil {
			return nil, closeErr
		}
	}
	for i, j := 0, len(matches)-1; i < j; i, j = i+1, j-1 {
		matches[i], matches[j] = matches[j], matches[i]
	}
	return matches, nil
}

func matchesRecord(r Record, opts QueryOptions) bool {
	if opts.Level != "" && !strings.EqualFold(r.Level, opts.Level) {
		return false
	}
	if opts.JobID != "" && r.JobID != opts.JobID {
		return false
	}
	if opts.Event != "" && !strings.EqualFold(r.Event, opts.Event) {
		return false
	}
	if opts.Query != "" {
		needle := strings.ToLower(opts.Query)
		blob, _ := json.Marshal(r)
		if !strings.Contains(strings.ToLower(string(blob)), needle) {
			return false
		}
	}
	return true
}
