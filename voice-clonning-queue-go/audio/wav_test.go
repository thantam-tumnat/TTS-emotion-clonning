package audio

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func writeWAV(t *testing.T, dir, name string, samples []int16, rate uint32) string {
	t.Helper()
	wav, err := EncodeWAV(samples, rate)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, wav, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestConcatWAVFilesJoinsEveryChunk(t *testing.T) {
	dir := t.TempDir()
	a := writeWAV(t, dir, "job_000.wav", []int16{1, 2, 3}, 48000)
	b := writeWAV(t, dir, "job_001.wav", []int16{4, 5}, 48000)

	wav, err := ConcatWAVFiles([]string{a, b})
	if err != nil {
		t.Fatalf("ConcatWAVFiles: %v", err)
	}
	want, _ := EncodeWAV([]int16{1, 2, 3, 4, 5}, 48000)
	if !bytes.Equal(wav, want) {
		t.Fatalf("joined wav differs from a single 5-sample wav:\n got %x\nwant %x", wav, want)
	}
}

func TestConcatWAVFilesRejectsMismatchedFormat(t *testing.T) {
	dir := t.TempDir()
	a := writeWAV(t, dir, "a.wav", []int16{1}, 48000)
	b := writeWAV(t, dir, "b.wav", []int16{2}, 24000)
	if _, err := ConcatWAVFiles([]string{a, b}); err == nil {
		t.Fatal("expected an error joining 48 kHz and 24 kHz audio")
	}
}

func TestPlayableWAVFallsBackToFirstChunk(t *testing.T) {
	dir := t.TempDir()
	a := writeWAV(t, dir, "a.wav", []int16{1}, 48000)
	b := writeWAV(t, dir, "b.wav", []int16{2}, 24000)
	first, _ := os.ReadFile(a)

	got, err := PlayableWAV([]interface{}{a, b})
	if err != nil {
		t.Fatalf("PlayableWAV: %v", err)
	}
	if !bytes.Equal(got, first) {
		t.Fatal("unjoinable chunks should fall back to the first chunk, as before")
	}
}

func TestPlayableWAVSkipsNonStringEntries(t *testing.T) {
	dir := t.TempDir()
	a := writeWAV(t, dir, "a.wav", []int16{7, 8}, 48000)
	got, err := PlayableWAV([]interface{}{nil, a, ""})
	if err != nil {
		t.Fatalf("PlayableWAV: %v", err)
	}
	want, _ := os.ReadFile(a)
	if !bytes.Equal(got, want) {
		t.Fatal("expected the single real path to be served as-is")
	}
}

func TestEncodeWAV(t *testing.T) {
	samples := []int16{0, 1000, -1000, 2000, -2000}
	sampleRate := uint32(48000)

	wav, err := EncodeWAV(samples, sampleRate)
	if err != nil {
		t.Fatalf("EncodeWAV failed: %v", err)
	}

	if len(wav) != 44+len(samples)*2 {
		t.Errorf("expected wav len %d, got %d", 44+len(samples)*2, len(wav))
	}

	if string(wav[:4]) != "RIFF" {
		t.Errorf("expected RIFF header, got %s", string(wav[:4]))
	}

	if string(wav[8:12]) != "WAVE" {
		t.Errorf("expected WAVE header, got %s", string(wav[8:12]))
	}
}

func TestNPZToWAV(t *testing.T) {
	// Create mock NPZ in memory
	buf := new(bytes.Buffer)
	zw := zip.NewWriter(buf)

	// Add chunk_000.npy with float32 samples [0.0, 0.5, -0.5]
	f, err := zw.Create("chunk_000.npy")
	if err != nil {
		t.Fatalf("failed to create zip entry: %v", err)
	}

	header := make([]byte, 10)
	header[0] = 0x93
	copy(header[1:6], "NUMPY")
	header[6] = 1
	header[7] = 0
	binary.LittleEndian.PutUint16(header[8:10], 0) // header_len = 0

	samplesFloats := []float32{0.0, 0.5, -0.5}
	sampleBytes := new(bytes.Buffer)
	for _, sf := range samplesFloats {
		binary.Write(sampleBytes, binary.LittleEndian, sf)
	}

	f.Write(header)
	f.Write(sampleBytes.Bytes())
	zw.Close()

	wavBytes, err := NPZToWAV(buf.Bytes())
	if err != nil {
		t.Fatalf("NPZToWAV failed: %v", err)
	}

	if len(wavBytes) != 44+len(samplesFloats)*2 {
		t.Errorf("expected wav len %d, got %d", 44+len(samplesFloats)*2, len(wavBytes))
	}
}
