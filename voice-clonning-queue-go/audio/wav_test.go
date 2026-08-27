package audio

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"testing"
)

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
