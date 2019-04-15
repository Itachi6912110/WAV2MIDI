import pretty_midi
import numpy as np
import sys

est_file_name = str(sys.argv[1])
out_file_name = str(sys.argv[2])

try:
    myfile = open(est_file_name, 'r')
except IOError:
    print("Could not open file ", est_file_name)
    exit()

print("Transcription: Generating MIDI to ", out_file_name)

est = np.loadtxt(est_file_name)
t1 = est[:,0].reshape((est.shape[0],))
t2 = est[:,1].reshape((est.shape[0],))
f  = est[:,2].reshape((est.shape[0],))

# Create a PrettyMIDI object
piano_chord = pretty_midi.PrettyMIDI()
# Create an Instrument instance for a piano instrument
piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
piano = pretty_midi.Instrument(program=piano_program)
# Iterate over note names, which will be converted to note number later
for f_idx in range(f.shape[0]):
    # Retrieve the MIDI note number for this note name
    note_pitch = int(round(pretty_midi.hz_to_note_number(f[f_idx])))
    if note_pitch > 127 or note_pitch < 0:
        print(note_pitch)
        input("OMG !!")
        note_pitch = 0 if note_pitch < 0 else 127
    #note_number = pretty_midi.note_name_to_number(note_name)
    # Create a Note instance, starting at 0s and ending at .5s
    note = pretty_midi.Note(
        velocity=100, pitch=note_pitch, start=t1[f_idx], end=t2[f_idx])
    # Add it to our cello instrument
    piano.notes.append(note)
# Add the cello instrument to the PrettyMIDI object
piano_chord.instruments.append(piano)
# Write out the MIDI data
piano_chord.write(out_file_name)