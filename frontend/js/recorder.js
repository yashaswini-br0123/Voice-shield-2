class VoiceRecorder {
    constructor(visualizer, onRecordComplete) {
        this.visualizer = visualizer;
        this.onRecordComplete = onRecordComplete;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.stream = null;
        this.timerInterval = null;
        this.secondsRecorded = 0;
        this.isRecording = false;

        this.btnStart = document.getElementById('btn-start-record');
        this.btnStop = document.getElementById('btn-stop-record');
        this.timerEl = document.getElementById('record-timer');

        this._initListeners();
    }

    _initListeners() {
        if (this.btnStart) {
            this.btnStart.addEventListener('click', () => this.startRecording());
        }
        if (this.btnStop) {
            this.btnStop.addEventListener('click', () => this.stopRecording());
        }
    }

    async startRecording() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.audioChunks = [];
            this.mediaRecorder = new MediaRecorder(this.stream);

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = async () => {
                const rawBlob = new Blob(this.audioChunks, { type: this.mediaRecorder.mimeType || 'audio/webm' });
                let finalBlob = rawBlob;
                let finalFileName = `recorded_voice_${Date.now()}.wav`;

                try {
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    const arrayBuffer = await rawBlob.arrayBuffer();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    finalBlob = this._audioBufferToWav(audioBuffer);
                    audioContext.close();
                } catch (e) {
                    console.warn("WAV conversion fallback:", e);
                }

                const recordedFile = new File([finalBlob], finalFileName, { type: 'audio/wav' });
                if (this.onRecordComplete) {
                    this.onRecordComplete(recordedFile, finalBlob);
                }
                this._cleanupStream();
            };

            this.mediaRecorder.start();
            this.isRecording = true;

            if (this.visualizer) {
                this.visualizer.start(this.stream);
            }
            this._startTimer();

            this.btnStart.classList.add('hidden');
            this.btnStop.classList.remove('hidden');

        } catch (err) {
            alert('Microphone access denied or unsupported: ' + err.message);
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this._stopTimer();
            if (this.visualizer) {
                this.visualizer.stop();
            }

            this.btnStop.classList.add('hidden');
            this.btnStart.classList.remove('hidden');
        }
    }

    _startTimer() {
        this.secondsRecorded = 0;
        this._updateTimerDisplay();
        this.timerInterval = setInterval(() => {
            this.secondsRecorded++;
            this._updateTimerDisplay();
        }, 1000);
    }

    _stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    _updateTimerDisplay() {
        if (this.timerEl) {
            const mins = String(Math.floor(this.secondsRecorded / 60)).padStart(2, '0');
            const secs = String(this.secondsRecorded % 60).padStart(2, '0');
            this.timerEl.textContent = `${mins}:${secs}`;
        }
    }

    _cleanupStream() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
    }

    _audioBufferToWav(buffer) {
        let numChannels = buffer.numberOfChannels;
        let sampleRate = buffer.sampleRate;
        let format = 1;
        let bitDepth = 16;
        let channels = [];
        for (let i = 0; i < numChannels; i++) {
            channels.push(buffer.getChannelData(i));
        }
        let numSamples = buffer.length;
        let dataSize = numSamples * numChannels * 2;
        let arrayBuffer = new ArrayBuffer(44 + dataSize);
        let view = new DataView(arrayBuffer);

        function writeString(offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * numChannels * 2, true);
        view.setUint16(32, numChannels * 2, true);
        view.setUint16(34, bitDepth, true);
        writeString(36, 'data');
        view.setUint32(40, dataSize, true);

        let offset = 44;
        for (let i = 0; i < numSamples; i++) {
            for (let ch = 0; ch < numChannels; ch++) {
                let s = Math.max(-1, Math.min(1, channels[ch][i]));
                view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                offset += 2;
            }
        }
        return new Blob([view], { type: 'audio/wav' });
    }
}
