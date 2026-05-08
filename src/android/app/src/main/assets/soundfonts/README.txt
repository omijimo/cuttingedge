Real-time accompaniment now uses Standard MIDI clip playback via Android's
platform synth (MediaPlayer/Sonivox), so a SoundFont asset is not required.

If you later add a custom SoundFont renderer, place the default file at:
  src/android/app/src/main/assets/soundfonts/default.sf2
