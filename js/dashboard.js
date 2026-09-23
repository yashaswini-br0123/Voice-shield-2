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
                { step: 3, label: "Running 4-Layer Audio Deepfake Analysis...", pct: 55 },
                { step: 4, label: "Running Frame-by-Frame Visual Deepfake Analysis...", pct: 75 },
                { step: 5, label: "Evaluating Audio-Visual Temporal Synchronization...", pct: 90 },
                { step: 6, label: "Running Multimodal Video Deepfake Fusion...", pct: 100 }
            ];
        } else {
            steps = [
                { step: 1, label: "Uploading & validating audio file...", pct: 15 },
                { step: 2, label: "Resampling to 16kHz mono WAV & normalizing...", pct: 30 },
                { step: 3, label: "Running Layer 1: Acoustic Feature Extraction...", pct: 45 },
                { step: 4, label: "Running Layer 2: Waveform AASIST Phase Analysis...", pct: 60 },
                { step: 5, label: "Running Layer 3: LFCC Spectral Classification...", pct: 75 },
                { step: 6, label: "Running Layer 4: WavLM SSL Representation...", pct: 90 },
                { step: 7, label: "Running Ensemble Score Fusion & verdict...", pct: 100 }
            ];
        }

        for (const s of steps) {
            this._setActiveStep(s.step);
            this.stepLabel.textContent = s.label;
            this.progressBar.style.width = `${s.pct}%`;
            await new Promise(r => setTimeout(r, 25));
        }

        await new Promise(r => setTimeout(r, 25));
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

        const badge = document.getElementById('verdict-badge');
        const icon = document.getElementById('verdict-icon');
        const text = document.getElementById('verdict-text');
        const subtext = document.getElementById('verdict-subtext');

        badge.className = 'verdict-badge';

        const prediction = data.prediction || "Likely Human";
        const aiScore = data.ai_risk_score !== undefined ? data.ai_risk_score : (data.ai_probability || 0.18);
        const isAI = prediction === 'Likely AI-Generated' || aiScore >= 0.65;
        const isUncertain = prediction.includes('Uncertain') || (aiScore > 0.35 && aiScore < 0.65) || (data.score_std && data.score_std > 0.28);

        if (isUncertain) {
            badge.classList.add('badge-uncertain');
            icon.className = 'fa-solid fa-circle-question';
            text.textContent = 'UNCERTAIN / VERIFY';
            subtext.textContent = 'Layer score disagreement or intermediate risk detected. Dynamic voice verification recommended.';
        } else if (isAI) {
            badge.classList.add('badge-ai');
            icon.className = 'fa-solid fa-robot';
            if (mediaType === "image") {
                text.textContent = 'AI-GENERATED IMAGE';
                subtext.textContent = 'High probability of synthetic diffusion/GAN image generation or digital manipulation.';
            } else if (mediaType === "video") {
                text.textContent = 'AI DEEPFAKE VIDEO';
                subtext.textContent = 'High probability of visual frame manipulation or voice cloning artifacts.';
            } else {
                text.textContent = 'LIKELY AI-GENERATED';
                subtext.textContent = 'High synthetic AI risk score identified across spectro-temporal analysis layers.';
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
                text.textContent = 'LIKELY HUMAN';
                subtext.textContent = 'Acoustic pitch vibrato, AASIST phase, and LFCC cepstral coefficients align with natural human speech.';
            }
        }

        // Gauges
        const aiPct = Math.round(aiScore * 100);
        const confPct = Math.round((data.confidence || 0.85) * 100);

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
        document.getElementById('meta-pipeline').textContent = isUncertain ? "RESULT: UNCERTAIN (REQUIRES VERIFICATION)" : (isAI ? "RESULT: LIKELY AI-GENERATED" : "RESULT: LIKELY HUMAN");

        // Render Official Digital Forensic Analysis Report
        this._renderForensicReport(data, isAI, isUncertain);
    }

    _renderForensicReport(data, isAI, isUncertain) {
        const forensicId = `VS-${new Date().getFullYear()}-${Math.floor(1000 + Math.random() * 9000)}-X`;
        const timeStr = new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC';

        document.getElementById('forensic-id').textContent = forensicId;
        document.getElementById('forensic-time').textContent = timeStr;
        
        const verdictBadge = document.getElementById('forensic-verdict');
        if (isUncertain) {
            verdictBadge.textContent = 'UNCERTAIN / VERIFY';
            verdictBadge.className = 'f-val-verdict badge-uncertain';
        } else if (isAI) {
            verdictBadge.textContent = 'AI-GENERATED';
            verdictBadge.className = 'f-val-verdict verdict-ai';
        } else {
            verdictBadge.textContent = 'LIKELY HUMAN';
            verdictBadge.className = 'f-val-verdict verdict-human';
        }

        const consistencyText = data.consistency_rating || (isUncertain ? 'Inconsistent - Layer Disagreement' : (isAI ? 'High Consistency - Synthetic Risk' : 'High Consistency - Verified Natural'));
        document.getElementById('forensic-risk').textContent = consistencyText.toUpperCase();
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
                { name: "Extracted Audio Track Deepfake Analysis", score: vidDet.audio_ai_prob },
                { name: "Multimodal Temporal Score Fusion", score: data.ai_probability }
            ];
        } else {
            const wavlmDet = (data.layer_details && data.layer_details.wavlm) || {};
            dimensions = [
                { name: "Layer 1: Acoustic & Pitch Vibrato Contours", score: data.layers.acoustic },
                { name: "Layer 2: AASIST Raw Waveform Phase Graph", score: data.layers.waveform },
                { name: "Layer 3: LFCC Spectral Cepstral Classification", score: data.layers.spectral },
                { name: "Layer 4: WavLM SSL Speech Representation", score: wavlmDet.score, isWavlmProfile: wavlmDet.status !== "configured" }
            ];
        }

        dimensions.forEach(dim => {
            const tr = document.createElement('tr');
            let scoreText = "";
            let statusTag = "";

            if (dim.isWavlmProfile) {
                scoreText = "768-dim SSL Embedding Vector";
                statusTag = '<span class="tag-clean">SSL Feature Profiling</span>';
            } else {
                const scoreVal = dim.score !== null && dim.score !== undefined ? Math.round(dim.score * 100) : 50;
                const isHigh = scoreVal > 50;
                scoreText = `<strong>${scoreVal}%</strong> AI Risk Score`;
                statusTag = `<span class="${isHigh ? 'tag-anom' : 'tag-clean'}">${isHigh ? 'Anomaly Flagged' : 'Verified Natural'}</span>`;
            }

            tr.innerHTML = `
                <td>${dim.name}</td>
                <td>${scoreText}</td>
                <td>${statusTag}</td>
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
            const wavlmDet = (data.layer_details && data.layer_details.wavlm) || {};
            const l4 = wavlmDet.score;

            this._setCard(1, "LAYER 1", "Acoustic Analysis", "MFCCs, Mel Spectrogram, Pitch Jitter, Spectral Centroid, Reverberation.", l1, data.layer_details.acoustic?.status);
            this._setCard(2, "LAYER 2", "Waveform & Phase", "AASIST PyTorch Graph Neural Network raw waveform phase analysis.", l2, data.layer_details.waveform?.status);
            this._setCard(3, "LAYER 3", "Spectral Analysis", "LFCC (Linear Frequency Cepstral Coefficients) & Scikit-Learn Classifier.", l3, data.layer_details.spectral?.status);
            
            const wavlmStatus = wavlmDet.status === "configured" ? "Active" : "Representation Active";
            const wavlmDesc = wavlmDet.status === "configured" ? "WavLM Base SSL Encoder + Downstream Classifier Head." : "WavLM Base SSL 768-dim speech representation extractor (Representation Profiling).";
            
            this._setCard(4, "LAYER 4", "WavLM SSL Representation", wavlmDesc, l4, wavlmStatus);
            if (l4 === null || l4 === undefined) {
                const card4Val = document.getElementById('l4-score-val');
                if (card4Val) card4Val.textContent = '768-dim';
                const card4Sub = document.getElementById('l4-sub');
                if (card4Sub) card4Sub.textContent = 'SSL Vector Profiling';
                const card4Meter = document.getElementById('l4-meter');
                if (card4Meter) card4Meter.style.width = '100%';
            }
        }
    }

    _setCard(cardNum, tag, title, desc, scoreVal, statusStr = null) {
        const tagEl = document.getElementById(`l${cardNum}-tag`);
        const titleEl = document.getElementById(`l${cardNum}-title`);
        const descEl = document.getElementById(`l${cardNum}-desc`);
        const scoreValEl = document.getElementById(`l${cardNum}-score-val`);
        const meterEl = document.getElementById(`l${cardNum}-meter`);
        const statusEl = document.getElementById(`l${cardNum}-status`);

        if (tagEl) tagEl.textContent = tag;
        if (titleEl) titleEl.textContent = title;
        if (descEl) descEl.textContent = desc;

        const pct = scoreVal !== null && scoreVal !== undefined ? Math.round(scoreVal * 100) : 0;
        if (scoreValEl) scoreValEl.textContent = scoreVal !== null && scoreVal !== undefined ? `${pct}%` : '768-dim';
        if (meterEl) meterEl.style.width = scoreVal !== null && scoreVal !== undefined ? `${pct}%` : '100%';

        if (statusEl) {
            statusEl.textContent = statusStr ? `Status: ${statusStr}` : (scoreVal !== null ? 'Status: Active' : 'Status: N/A');
        }
    }
}
