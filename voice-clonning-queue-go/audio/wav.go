package audio

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
)

// NPZToWAV parses an NPZ archive (zip of float32 .npy chunks) and returns concatenated 16-bit PCM WAV bytes.
func NPZToWAV(npzBytes []byte) ([]byte, error) {
	if len(npzBytes) == 0 {
		return nil, fmt.Errorf("empty payload")
	}

	zr, err := zip.NewReader(bytes.NewReader(npzBytes), int64(len(npzBytes)))
	if err != nil {
		return nil, fmt.Errorf("invalid zip: %w", err)
	}

	sampleRate := uint32(48000)
	var chunkFiles []*zip.File

	for _, f := range zr.File {
		if strings.HasPrefix(f.Name, "sample_rate") {
			rc, err := f.Open()
			if err == nil {
				b, _ := io.ReadAll(rc)
				rc.Close()
				sr := parseNpyScalarInt(b)
				if sr > 0 {
					sampleRate = uint32(sr)
				}
			}
		} else if strings.HasPrefix(f.Name, "chunk_") && strings.HasSuffix(f.Name, ".npy") {
			chunkFiles = append(chunkFiles, f)
		}
	}

	// Sort chunks numerically: chunk_000, chunk_001, etc.
	sort.Slice(chunkFiles, func(i, j int) bool {
		return chunkFiles[i].Name < chunkFiles[j].Name
	})

	var allSamples []int16
	for _, f := range chunkFiles {
		rc, err := f.Open()
		if err != nil {
			continue
		}
		b, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			continue
		}
		samples := parseNpyFloat32ToPCM16(b)
		allSamples = append(allSamples, samples...)
	}

	if len(allSamples) == 0 {
		return nil, fmt.Errorf("no audio samples found in NPZ")
	}

	return EncodeWAV(allSamples, sampleRate)
}

// ReadWAVFile reads a WAV file from disk.
func ReadWAVFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}

// ConcatWAVFiles joins PCM WAV files end to end into one WAV, in the order given.
//
// A "files"-mode job writes one WAV per chunk, and a caller that previews only the
// first one plays a fraction of the take while looking complete -- the dashboard did
// exactly that, offering chunk 1 of a two-chunk script as the whole job. The chunks
// come from one render, so they share a format; a file that does not is an error
// rather than something to resample, because splicing mismatched PCM is noise.
func ConcatWAVFiles(paths []string) ([]byte, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("no wav files")
	}
	var fmtChunk []byte
	var data bytes.Buffer
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		f, d, err := splitWAV(raw)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", p, err)
		}
		if fmtChunk == nil {
			fmtChunk = f
		} else if !bytes.Equal(fmtChunk, f) {
			return nil, fmt.Errorf("%s: format differs from %s", p, paths[0])
		}
		data.Write(d)
	}

	out := new(bytes.Buffer)
	out.Grow(20 + len(fmtChunk) + data.Len())
	out.WriteString("RIFF")
	binary.Write(out, binary.LittleEndian, uint32(4+8+len(fmtChunk)+8+data.Len()))
	out.WriteString("WAVE")
	out.WriteString("fmt ")
	binary.Write(out, binary.LittleEndian, uint32(len(fmtChunk)))
	out.Write(fmtChunk)
	out.WriteString("data")
	binary.Write(out, binary.LittleEndian, uint32(data.Len()))
	out.Write(data.Bytes())
	return out.Bytes(), nil
}

// PlayableWAV turns a files-mode result's "files" list into one WAV: every chunk
// joined in order, or -- if they cannot be joined -- the first chunk alone, which is
// all that was served before, so a join failure never costs the preview entirely.
func PlayableWAV(files []interface{}) ([]byte, error) {
	paths := make([]string, 0, len(files))
	for _, f := range files {
		if p, ok := f.(string); ok && p != "" {
			paths = append(paths, p)
		}
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("no wav files")
	}
	if len(paths) > 1 {
		if wav, err := ConcatWAVFiles(paths); err == nil {
			return wav, nil
		}
	}
	return ReadWAVFile(paths[0])
}

// splitWAV returns the body of a WAV's "fmt " and "data" chunks, skipping any
// others (LIST, fact, ...) rather than assuming the canonical 44-byte layout.
func splitWAV(raw []byte) (fmtBody, dataBody []byte, err error) {
	if len(raw) < 12 || string(raw[:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, nil, fmt.Errorf("not a RIFF/WAVE file")
	}
	for pos := 12; pos+8 <= len(raw); {
		id := string(raw[pos : pos+4])
		size := int(binary.LittleEndian.Uint32(raw[pos+4 : pos+8]))
		body := pos + 8
		end := body + size
		if end > len(raw) {
			if id != "data" {
				return nil, nil, fmt.Errorf("truncated %q chunk", id)
			}
			end = len(raw) // a writer killed mid-file leaves a short data chunk
		}
		switch id {
		case "fmt ":
			fmtBody = raw[body:end]
		case "data":
			dataBody = raw[body:end]
		}
		pos = end + size%2 // chunks are word-aligned
	}
	if fmtBody == nil || dataBody == nil {
		return nil, nil, fmt.Errorf("missing fmt or data chunk")
	}
	return fmtBody, dataBody, nil
}

func parseNpyScalarInt(b []byte) int {
	if len(b) < 10 {
		return 0
	}
	headerLen := binary.LittleEndian.Uint16(b[8:10])
	dataOffset := 10 + int(headerLen)
	if len(b) >= dataOffset+8 {
		return int(binary.LittleEndian.Uint64(b[dataOffset : dataOffset+8]))
	} else if len(b) >= dataOffset+4 {
		return int(binary.LittleEndian.Uint32(b[dataOffset : dataOffset+4]))
	}
	return 0
}

func parseNpyFloat32ToPCM16(b []byte) []int16 {
	if len(b) < 10 {
		return nil
	}
	headerLen := binary.LittleEndian.Uint16(b[8:10])
	dataOffset := 10 + int(headerLen)
	if dataOffset > len(b) {
		return nil
	}
	rawData := b[dataOffset:]
	numFloats := len(rawData) / 4
	samples := make([]int16, numFloats)

	for i := 0; i < numFloats; i++ {
		bits := binary.LittleEndian.Uint32(rawData[i*4 : (i+1)*4])
		f := math.Float32frombits(bits)
		if f > 1.0 {
			f = 1.0
		} else if f < -1.0 {
			f = -1.0
		}
		samples[i] = int16(f * 32767.0)
	}
	return samples
}

// EncodeWAV creates a standard 16-bit Mono WAV byte slice.
func EncodeWAV(samples []int16, sampleRate uint32) ([]byte, error) {
	numSamples := len(samples)
	dataSize := uint32(numSamples * 2)
	fileSize := 36 + dataSize

	buf := new(bytes.Buffer)
	buf.Grow(int(44 + dataSize))

	// RIFF Header
	buf.WriteString("RIFF")
	binary.Write(buf, binary.LittleEndian, fileSize)
	buf.WriteString("WAVE")

	// fmt chunk
	buf.WriteString("fmt ")
	binary.Write(buf, binary.LittleEndian, uint32(16))   // Subchunk1Size (16 for PCM)
	binary.Write(buf, binary.LittleEndian, uint16(1))    // AudioFormat (1 = PCM)
	binary.Write(buf, binary.LittleEndian, uint16(1))    // NumChannels (1 = Mono)
	binary.Write(buf, binary.LittleEndian, sampleRate)   // SampleRate
	binary.Write(buf, binary.LittleEndian, sampleRate*2) // ByteRate (SampleRate * 1 * 16/8)
	binary.Write(buf, binary.LittleEndian, uint16(2))    // BlockAlign (1 * 16/8)
	binary.Write(buf, binary.LittleEndian, uint16(16))   // BitsPerSample

	// data chunk
	buf.WriteString("data")
	binary.Write(buf, binary.LittleEndian, dataSize)

	for _, s := range samples {
		binary.Write(buf, binary.LittleEndian, s)
	}

	return buf.Bytes(), nil
}
