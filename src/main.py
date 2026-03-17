import mido
from collections import defaultdict
import glob
 
# ──────────────────────────────────────────────
# Quantization helpers
# ──────────────────────────────────────────────
 
def quantize_velocity(velocity, num_bins=32):
    if velocity == 0:
        return 0
    return max(1, round(velocity / 127 * (num_bins - 1)))
 
 
def dequantize_velocity(vel_bin, num_bins=32):
    if vel_bin == 0:
        return 0
    return round(vel_bin / (num_bins - 1) * 127)
 
 
def quantize_time(ticks, ticks_per_beat, num_steps=32):
    step_size = ticks_per_beat / (num_steps / 4)
    return round(ticks / step_size)
 
 
def dequantize_time(steps, ticks_per_beat, num_steps=32):
    step_size = ticks_per_beat / (num_steps / 4)
    return round(steps * step_size)
 
 
# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────
 
def _extract_notes(mid):
    ticks_per_beat = mid.ticks_per_beat
    notes = []
 
    for track in mid.tracks:
        absolute_time = 0
        open_notes = defaultdict(list)
 
        for msg in track:
            absolute_time += msg.time
 
            if msg.type == 'note_on' and msg.velocity > 0:
                open_notes[msg.note].append((absolute_time, msg.velocity))
 
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if open_notes[msg.note]:
                    start_time, velocity = open_notes[msg.note].pop(0)
                    duration = absolute_time - start_time
                    notes.append({
                        'pitch': msg.note,
                        'velocity': velocity,
                        'start': start_time,
                        'duration': duration
                    })
 
    notes.sort(key=lambda n: (n['start'], n['pitch']))
    return notes, ticks_per_beat
 
 
def _parse_token(token):
    parts = token.split('_')
 
    if parts[0] == 'REST':
        time_val = int(parts[1][1:])
        return {'type': 'rest', 'time_shift': time_val}
 
    else:
        pitch = int(parts[0][1:])
        velocity = int(parts[1][1:])
        duration = int(parts[2][1:])
        return {
            'type': 'note',
            'pitch': pitch,
            'velocity': velocity,
            'duration': duration
        }
 
 
# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
 
def tokenize_midi(filepath):
    """Read a MIDI file and return a list of token strings."""
    mid = mido.MidiFile(filepath)
    notes, tpb = _extract_notes(mid)
    tokens = []
    prev_start = 0
 
    for note in notes:
        time_shift = quantize_time(note['start'] - prev_start, tpb)
 
        if time_shift > 0:
            tokens.append(f"REST_T{time_shift}")
 
        vel = quantize_velocity(note['velocity'])
        dur = quantize_time(note['duration'], tpb)
        dur = max(1, dur)
 
        tokens.append(f"P{note['pitch']}_V{vel}_D{dur}")
        prev_start = note['start']
 
    return tokens
 
 
def detokenize_midi(tokens, output_path, ticks_per_beat=480):
    """Convert a list of token strings into a playable MIDI file."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
 
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))
 
    current_time = 0
    note_events = []
 
    for token in tokens:
        parsed = _parse_token(token)
 
        if parsed['type'] == 'rest':
            time_shift = dequantize_time(parsed['time_shift'], ticks_per_beat)
            current_time += time_shift
 
        elif parsed['type'] == 'note':
            velocity = dequantize_velocity(parsed['velocity'])
            duration = dequantize_time(parsed['duration'], ticks_per_beat)
            duration = max(1, duration)
 
            note_events.append({
                'time': current_time,
                'type': 'note_on',
                'note': parsed['pitch'],
                'velocity': velocity
            })
            note_events.append({
                'time': current_time + duration,
                'type': 'note_off',
                'note': parsed['pitch'],
                'velocity': 0
            })
 
    note_events.sort(key=lambda e: (e['time'], e['type'] == 'note_on'))
 
    prev_time = 0
    for event in note_events:
        delta = event['time'] - prev_time
 
        if event['type'] == 'note_on':
            track.append(mido.Message(
                'note_on',
                note=event['note'],
                velocity=event['velocity'],
                time=delta
            ))
        else:
            track.append(mido.Message(
                'note_off',
                note=event['note'],
                velocity=0,
                time=delta
            ))
 
        prev_time = event['time']
 
    mid.save(output_path)
    return output_path
 
 
# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
 
if __name__ == '__main__':
    
    midi_files = glob.glob("data/maestro/**/*.midi", recursive=True)
    tokenized = tokenize_midi(midi_files[10])
    detokenized = detokenize_midi(tokenized, "output.mid")

