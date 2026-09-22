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
	text    *os.File
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
		if l.text != nil {
			_ = l.text.Close()
		}
		name := filepath.Join(l.dir, fmt.Sprintf("%s-%s.jsonl", safePart(l.service), date))
		f, err := os.OpenFile(name, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		l.file = f
		textName := filepath.Join(l.dir, fmt.Sprintf("%s-%s.log", safePart(l.service), date))
		textFile, err := os.OpenFile(textName, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			_ = f.Close()
			l.file = nil
			return err
		}
		l.text = textFile
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
	plain := fmt.Sprintf("[%s] %-5s %-18s", record.Time, record.Level, record.Event)
	if jobID != "" {
		plain += " job=" + jobID
	}
	if message != "" {
		plain += " " + strings.ReplaceAll(strings.ReplaceAll(message, "\r", " "), "\n", "\\n")
	}
	if len(fields) > 0 {
		fieldBytes, _ := json.Marshal(fields)
		plain += " " + string(fieldBytes)
	}
	if _, err = l.text.WriteString(plain + "\n"); err != nil {
		return err
	}
	if syncDisk {
		if err := l.file.Sync(); err != nil {
			return err
		}
		return l.text.Sync()
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
	textErr := l.text.Sync()
	textCloseErr := l.text.Close()
	l.file = nil
	l.text = nil
	return errors.Join(err, closeErr, textErr, textCloseErr)
}

func safePart(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "service"
	}
	return regexp.MustCompile(`[^a-zA-Z0-9._-]+`).ReplaceAllString(value, "_")
}

type QueryOptions struct {
	Date        string
	Service     string
	Level       string
	JobID       string
	Event       string
	Query       string
	Limit       int
	Before      string // return records older than this RFC3339 timestamp
	After       string // return records newer than this RFC3339 timestamp
	ExcludeHTTP bool
}

func Query(dir string, opts QueryOptions) ([]Record, error) {
	page, err := QueryPage(dir, opts)
	return page.Records, err
}

type QueryResult struct {
	Records  []Record
	HasOlder bool
	HasNewer bool
}

func QueryPage(dir string, opts QueryOptions) (QueryResult, error) {
	if dir == "" {
		dir = "logs"
	}
	if opts.Date == "" {
		opts.Date = time.Now().Format("2006-01-02")
	}
	if !regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`).MatchString(opts.Date) {
		return QueryResult{}, fmt.Errorf("invalid date")
	}
	if opts.Limit <= 0 || opts.Limit > 1000 {
		opts.Limit = 200
	}
	before, err := parseCursor(opts.Before)
	if err != nil {
		return QueryResult{}, err
	}
	after, err := parseCursor(opts.After)
	if err != nil {
		return QueryResult{}, err
	}
	if before != nil && after != nil {
		return QueryResult{}, fmt.Errorf("before and after cannot be used together")
	}
	pattern := "*-" + opts.Date + ".jsonl"
	if opts.Service != "" {
		pattern = safePart(opts.Service) + "-" + opts.Date + ".jsonl"
	}
	files, err := filepath.Glob(filepath.Join(dir, pattern))
	if err != nil {
		return QueryResult{}, err
	}
	sort.Strings(files)
	var matches []Record
	for _, name := range files {
		f, err := os.Open(name)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return QueryResult{}, err
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
		}
		closeErr := f.Close()
		if err := scanner.Err(); err != nil {
			return QueryResult{}, err
		}
		if closeErr != nil {
			return QueryResult{}, closeErr
		}
	}
	sort.SliceStable(matches, func(i, j int) bool {
		return matches[i].Time < matches[j].Time
	})
	filtered := matches
	if before != nil || after != nil {
		filtered = make([]Record, 0, len(matches))
		for _, record := range matches {
			timestamp, parseErr := time.Parse(time.RFC3339Nano, record.Time)
			if parseErr != nil {
				continue
			}
			if before != nil && timestamp.Before(*before) {
				filtered = append(filtered, record)
			}
			if after != nil && timestamp.After(*after) {
				filtered = append(filtered, record)
			}
		}
	}
	if len(filtered) > opts.Limit {
		if after != nil {
			filtered = filtered[:opts.Limit]
		} else {
			filtered = filtered[len(filtered)-opts.Limit:]
		}
	}
	for i, j := 0, len(filtered)-1; i < j; i, j = i+1, j-1 {
		filtered[i], filtered[j] = filtered[j], filtered[i]
	}
	page := QueryResult{Records: filtered}
	if len(filtered) > 0 {
		newest := filtered[0].Time
		oldest := filtered[len(filtered)-1].Time
		for _, record := range matches {
			page.HasNewer = page.HasNewer || record.Time > newest
			page.HasOlder = page.HasOlder || record.Time < oldest
		}
	}
	return page, nil
}

func parseCursor(value string) (*time.Time, error) {
	if value == "" {
		return nil, nil
	}
	timestamp, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		return nil, fmt.Errorf("invalid cursor: %w", err)
	}
	return &timestamp, nil
}

func matchesRecord(r Record, opts QueryOptions) bool {
	if opts.ExcludeHTTP && strings.EqualFold(r.Event, "http_request") {
		return false
	}
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
