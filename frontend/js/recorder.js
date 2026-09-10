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

            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                const recordedFile = new File([audioBlob], `recorded_voice_${Date.now()}.wav`, { type: 'audio/wav' });
                if (this.onRecordComplete) {
                    this.onRecordComplete(recordedFile, audioBlob);
                }
                this._cleanupStream();
            };

            this.mediaRecorder.start();
            this.isRecording = true;

            // Start visualizer and timer
            if (this.visualizer) {
                this.visualizer.start(this.stream);
            }
            this._startTimer();

            // Toggle UI buttons
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
}
