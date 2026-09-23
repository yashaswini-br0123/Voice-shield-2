class ThemeManager {
    constructor() {
        this.themeToggleBtn = document.getElementById('theme-toggle');
        this.themeLabel = document.getElementById('theme-label');
        this.init();
    }

    init() {
        const savedTheme = localStorage.getItem('voiceshield_theme') || 'light';
        this.setTheme(savedTheme);

        if (this.themeToggleBtn) {
            this.themeToggleBtn.addEventListener('click', () => {
                const isDark = document.body.classList.contains('dark-theme');
                this.setTheme(isDark ? 'light' : 'dark');
            });
        }
    }

    setTheme(theme) {
        if (theme === 'dark') {
            document.body.classList.remove('light-theme');
            document.body.classList.add('dark-theme');
            if (this.themeLabel) this.themeLabel.textContent = 'Light Mode';
            if (this.themeToggleBtn) {
                const icon = this.themeToggleBtn.querySelector('i');
                if (icon) icon.className = 'fa-solid fa-sun';
            }
            localStorage.setItem('voiceshield_theme', 'dark');
        } else {
            document.body.classList.remove('dark-theme');
            document.body.classList.add('light-theme');
            if (this.themeLabel) this.themeLabel.textContent = 'Dark Mode';
            if (this.themeToggleBtn) {
                const icon = this.themeToggleBtn.querySelector('i');
                if (icon) icon.className = 'fa-solid fa-moon';
            }
            localStorage.setItem('voiceshield_theme', 'light');
        }
    }
}

class DashboardManager {
    constructor() {
        this.themeManager = new ThemeManager();
        this.welcomeDashboard = document.getElementById('welcome-dashboard');
        this.progressSection = document.getElementById('analysis-progress-section');
        this.errorDisplay = document.getElementById('error-display-card');
        this.errorMessageText = document.getElementById('error-message-text');
        this.progressBar = document.getElementById('progress-bar');
        this.stepLabel = document.getElementById('current-step-label');
        this.resultSection = document.getElementById('result-dashboard');
        this.demoBanner = document.getElementById('demo-banner');
    }

    async runProgressStepper(mediaType = "audio") {
        if (this.welcomeDashboard) this.welcomeDashboard.classList.add('hidden');
        if (this.resultSection) this.resultSection.classList.add('hidden');
        if (this.errorDisplay) this.errorDisplay.classList.add('hidden');
        this.progressSection.classList.remove('hidden');

        let steps = [];
        if (mediaType === "image") {
            steps = [
                { step: 1, label: "Uploading image file...", pct: 15 },
                { step: 2, label: "Normalizing image dimensions & color space...", pct: 35 },
                { step: 3, label: "Running 2D FFT Frequency Spectral Analysis...", pct: 55 },
                { step: 4, label: "Running ELA Compression Residual Check...", pct: 75 },
                { step: 5, label: "Running Noise Covariance & Facial Edge Analysis...", pct: 90 },
                { step: 6, label: "Synthesizing AI Image Deepfake Score...", pct: 100 }
            ];
        } else if (mediaType === "video") {
            steps = [
                { step: 1, label: "Uploading video container...", pct: 15 },
                { step: 2, label: "Sampling video keyframes & audio stream...", pct: 35 },
                { step: 3, label: "Running 3-Layer Audio Deepfake Analysis...", pct: 55 },
                { step: 4, label: "Running Frame-by-Frame Visual Deepfake Analysis...", pct: 75 },
                { step: 5, label: "Evaluating Audio-Visual Temporal Synchronization...", pct: 90 },
                { step: 6, label: "Running Multimodal Video Deepfake Fusion...", pct: 100 }
            ];
        } else {
            steps = [
                { step: 1, label: "Uploading & validating audio file...", pct: 15 },
                { step: 2, label: "Resampling to 16kHz mono WAV & normalizing...", pct: 35 },
                { step: 3, label: "Running Layer 1: Acoustic Feature Extraction...", pct: 55 },
                { step: 4, label: "Running Layer 2: Waveform AASIST Phase Analysis...", pct: 75 },
                { step: 5, label: "Running Layer 3: LFCC Spectral Classification...", pct: 90 },
                { step: 6, label: "Running Ensemble Score Fusion & verdict...", pct: 100 }
            ];
        }

        for (const s of steps) {
            this._setActiveStep(s.step);
            this.stepLabel.textContent = s.label;
            this.progressBar.style.width = `${s.pct}%`;
            await new Promise(r => setTimeout(r, 30));
        }

        await new Promise(r => setTimeout(r, 30));
        this.progressSection.classList.add('hidden');
    }

    showError(msg) {
        if (this.welcomeDashboard) this.welcomeDashboard.classList.add('hidden');
        if (this.progressSection) this.progressSection.classList.add('hidden');
        if (this.resultSection) this.resultSection.classList.add('hidden');
        
        if (this.errorDisplay) {
            this.errorMessageText.textContent = msg || "Failed to analyze media.";
            this.errorDisplay.classList.remove('hidden');
        }
    }

    _setActiveStep(stepNum) {
        document.querySelectorAll('.stepper-step').forEach(el => {
            const num = parseInt(el.getAttribute('data-step'));
            el.classList.remove('active', 'completed');
            if (num === stepNum) {
                el.classList.add('active');
            } else if (num < stepNum) {
                el.classList.add('completed');
            }
        });
    }

    renderResults(data) {
        if (this.welcomeDashboard) this.welcomeDashboard.classList.add('hidden');
        if (this.errorDisplay) this.errorDisplay.classList.add('hidden');
        
        this.resultSection.classList.remove('hidden');
        this.resultSection.scrollIntoView({ behavior: 'smooth' });

        if (data.demo_mode) {
            this.demoBanner.classList.remove('hidden');
        } else {
            this.demoBanner.classList.add('hidden');
        }

        const mediaType = data.media_type || "audio";
        document.getElementById('media-type-badge').textContent = `FINAL CLASSIFICATION: ${mediaType.toUpperCase()}`;

        // Primary Verdict: Definitive Binary Outcome (AI vs REAL/HUMAN)
        const badge = document.getElementById('verdict-badge');
        const icon = document.getElementById('verdict-icon');
        const text = document.getElementById('verdict-text');
        const subtext = document.getElementById('verdict-subtext');

        badge.className = 'verdict-badge';

        const isAI = data.prediction === 'Likely AI-Generated' || data.ai_probability > 0.50;

        if (isAI) {
            badge.classList.add('badge-ai');
            icon.className = 'fa-solid fa-robot';
            if (mediaType === "image") {
                text.textContent = 'AI-GENERATED IMAGE';
                subtext.textContent = 'High probability of synthetic diffusion/GAN image generation or digital manipulation.';
            } else if (mediaType === "video") {
                text.textContent = 'AI DEEPFAKE VIDEO';
                subtext.textContent = 'High probability of visual frame manipulation or voice cloning artifacts.';
            } else {
                text.textContent = 'AI-GENERATED VOICE';
                subtext.textContent = 'High probability of synthetic speech synthesis or voice conversion artifacts.';
            }

        } else {
            badge.classList.add('badge-human');
            icon.className = 'fa-solid fa-user-check';
            if (mediaType === "image") {
                text.textContent = 'REAL / AUTHENTIC IMAGE';
                subtext.textContent = 'Spatial noise covariance and ELA compression align with authentic camera photos.';
            } else if (mediaType === "video") {
                text.textContent = 'REAL / AUTHENTIC VIDEO';
                subtext.textContent = 'Visual keyframe gradients and audio sync align with authentic video recordings.';
            } else {
                text.textContent = 'REAL / HUMAN VOICE';
                subtext.textContent = 'Acoustic pitch vibrato, AASIST phase, and LFCC cepstral coefficients align with natural human speech.';
            }
        }

        // Gauges
        const aiPct = Math.round(data.ai_probability * 100);
        const confPct = Math.round(data.confidence * 100);

        document.getElementById('gauge-score').textContent = `${aiPct}%`;
        document.getElementById('gauge-fill').style.width = `${aiPct}%`;
        document.getElementById('confidence-score').textContent = `${confPct}%`;

        // Adapt Layer Cards
        this._updateLayerCards(data);

        // Technical Explanation
        document.getElementById('explanation-text').textContent = data.explanation;

        // Anomalies List
        const anomalyUl = document.getElementById('anomaly-list');
        anomalyUl.innerHTML = '';
        if (data.anomalies && data.anomalies.length > 0) {
            data.anomalies.forEach(anom => {
                const li = document.createElement('li');
                li.textContent = anom;
                anomalyUl.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No significant spectro-temporal or visual anomalies detected.';
            anomalyUl.appendChild(li);
        }

        // Technical Profile Metadata
        document.getElementById('meta-category').textContent = mediaType.toUpperCase();
        document.getElementById('meta-duration').textContent = data.audio ? `${data.audio.duration || 'N/A'}s` : (data.layer_details.image ? data.layer_details.image.dimensions : 'N/A');
        document.getElementById('meta-time').textContent = `${data.processing_time_sec}s`;
        document.getElementById('meta-pipeline').textContent = isAI ? "RESULT: AI-GENERATED" : "RESULT: REAL (HUMAN)";

        // Render Official Digital Forensic Analysis Report
        this._renderForensicReport(data, isAI);
    }

    _renderForensicReport(data, isAI) {
        const forensicId = `VS-${new Date().getFullYear()}-${Math.floor(1000 + Math.random() * 9000)}-X`;
        const timeStr = new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC';

        document.getElementById('forensic-id').textContent = forensicId;
        document.getElementById('forensic-time').textContent = timeStr;
        
        const verdictBadge = document.getElementById('forensic-verdict');
        verdictBadge.textContent = isAI ? 'AI-GENERATED' : 'HUMAN / REAL';
        verdictBadge.className = `f-val-verdict ${isAI ? 'verdict-ai' : 'verdict-human'}`;

        document.getElementById('forensic-risk').textContent = isAI ? 'HIGH SYNTHETIC RISK' : 'LOW SYNTHETIC RISK (VERIFIED REAL)';
        document.getElementById('forensic-summary-text').textContent = data.explanation;

        const tableBody = document.getElementById('forensic-table-body');
        tableBody.innerHTML = '';

        const mediaType = data.media_type || "audio";
        let dimensions = [];

        if (mediaType === "image") {
            const imgDet = data.layer_details.image || {};
            dimensions = [
                { name: "2D FFT Frequency Spectral Grid", score: imgDet.fft_spectral_score },
                { name: "Error Level Analysis (ELA) Compression", score: imgDet.ela_compression_score },
                { name: "Pixel Noise Covariance & Edge Consistency", score: imgDet.noise_covariance_score }
            ];
        } else if (mediaType === "video") {
            const vidDet = data.layer_details.video || {};
            dimensions = [
                { name: "Visual Keyframe Spatial Artifacts", score: vidDet.visual_ai_prob },
                { name: "Extracted Audio Track 3-Layer Deepfake", score: vidDet.audio_ai_prob },
                { name: "Multimodal Temporal Score Fusion", score: data.ai_probability }
            ];
        } else {
            dimensions = [
                { name: "Layer 1: Acoustic & Pitch Vibrato Contours", score: data.layers.acoustic },
                { name: "Layer 2: Waveform & AASIST Phase Graph", score: data.layers.waveform },
                { name: "Layer 3: LFCC Spectral Cepstral Classification", score: data.layers.spectral }
            ];
        }

        dimensions.forEach(dim => {
            const tr = document.createElement('tr');
            const scoreVal = dim.score !== null && dim.score !== undefined ? Math.round(dim.score * 100) : 50;
            const isHigh = scoreVal > 50;

            tr.innerHTML = `
                <td>${dim.name}</td>
                <td><strong>${scoreVal}%</strong> Synthetic Risk</td>
                <td><span class="${isHigh ? 'tag-anom' : 'tag-clean'}">${isHigh ? 'Anomaly Flagged' : 'Verified Natural'}</span></td>
            `;
            tableBody.appendChild(tr);
        });

        // Hook export button if not hooked
        const btnExport = document.getElementById('btn-export-report');
        if (btnExport && !btnExport.dataset.hooked) {
            btnExport.dataset.hooked = "true";
            btnExport.addEventListener('click', () => {
                window.print();
            });
        }
    }

    _updateLayerCards(data) {
        const mediaType = data.media_type || "audio";

        if (mediaType === "image") {
            const imgDet = data.layer_details.image || {};
            
            this._setCard(1, "FEATURE 1", "2D FFT Spectral", "Frequency spectrum grid ring analysis for diffusion/GAN artifacts.", imgDet.fft_spectral_score);
            this._setCard(2, "FEATURE 2", "ELA Compression", "Error Level Analysis JPEG compression residual variance check.", imgDet.ela_compression_score);
            this._setCard(3, "FEATURE 3", "Noise & Facial Edges", "Pixel noise covariance and facial boundary gradient consistency.", imgDet.noise_covariance_score);

        } else if (mediaType === "video") {
            const vidDet = data.layer_details.video || {};

            this._setCard(1, "COMPONENT 1", "Visual Keyframes", "Frame-by-frame 2D FFT & ELA keyframe analysis across sampled frames.", vidDet.visual_ai_prob);
            this._setCard(2, "COMPONENT 2", "Audio Track Deepfake", "3-Layer spectro-temporal audio deepfake analysis of extracted audio.", vidDet.audio_ai_prob);
            this._setCard(3, "COMPONENT 3", "Multimodal Fusion", "Combined Visual + Audio deepfake probability score.", data.ai_probability);

        } else {
            const l1 = data.layers.acoustic;
            const l2 = data.layers.waveform;
            const l3 = data.layers.spectral;

            this._setCard(1, "LAYER 1", "Acoustic Analysis", "MFCCs, Mel Spectrogram, Pitch Jitter, Spectral Centroid, Reverberation.", l1, data.layer_details.acoustic?.status);
            this._setCard(2, "LAYER 2", "Waveform & Phase", "AASIST PyTorch Graph Neural Network raw waveform phase analysis.", l2, data.layer_details.waveform?.status);
            this._setCard(3, "LAYER 3", "Spectral Analysis", "LFCC (Linear Frequency Cepstral Coefficients) & Scikit-Learn Classifier.", l3, data.layer_details.spectral?.status);
        }
    }

    _setCard(cardNum, tag, title, desc, scoreVal, statusStr = null) {
        document.getElementById(`l${cardNum}-tag`).textContent = tag;
        document.getElementById(`l${cardNum}-title`).textContent = title;
        document.getElementById(`l${cardNum}-desc`).textContent = desc;

        const pct = scoreVal !== null && scoreVal !== undefined ? Math.round(scoreVal * 100) : 0;
        document.getElementById(`l${cardNum}-score-val`).textContent = scoreVal !== null && scoreVal !== undefined ? `${pct}%` : 'N/A';
        document.getElementById(`l${cardNum}-meter`).style.width = `${pct}%`;

        if (statusStr) {
            document.getElementById(`l${cardNum}-status`).textContent = `Status: ${statusStr}`;
        } else {
            document.getElementById(`l${cardNum}-status`).textContent = scoreVal !== null ? 'Status: Active' : 'Status: N/A';
        }
    }
}
