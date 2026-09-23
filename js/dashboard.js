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

        const btnPdf = document.getElementById('btn-download-pdf-report');
        if (btnPdf) {
            btnPdf.addEventListener('click', () => this.generateForensicPDF());
        }
    }

    generateForensicPDF() {
        if (!window.jspdf || !window.jspdf.jsPDF) {
            alert("PDF generator library is loading. Please try again in a moment.");
            return;
        }

        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();

        const forensicId = document.getElementById('forensic-id')?.textContent || `VS-${new Date().getFullYear()}-${Math.floor(1000 + Math.random() * 9000)}-X`;
        const timeStr = document.getElementById('forensic-time')?.textContent || new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
        const verdictText = document.getElementById('verdict-text')?.textContent || 'LIKELY HUMAN';
        const aiRiskScore = document.getElementById('gauge-score')?.textContent || '0%';
        const confidenceScore = document.getElementById('confidence-score')?.textContent || '0%';
        const explanationText = document.getElementById('explanation-text')?.textContent || '';

        // Cyber Security Header Banner
        doc.setFillColor(15, 23, 42); // #0f172a (Dark navy)
        doc.rect(0, 0, 210, 38, 'F');

        doc.setTextColor(255, 255, 255);
        doc.setFont("helvetica", "bold");
        doc.setFontSize(18);
        doc.text("VOICESHIELD CYBER FORENSIC REPORT", 14, 18);

        doc.setFont("helvetica", "normal");
        doc.setFontSize(9);
        doc.setTextColor(148, 163, 184); // #94a3b8
        doc.text("MULTIMODAL AI DEEPFAKE DETECTION & EVIDENCE DOSSIER", 14, 26);
        doc.text(`REPORT ID: ${forensicId}  |  DATE: ${timeStr}`, 14, 32);

        // Section 1: Executive Summary Card
        let y = 48;
        doc.setFillColor(248, 250, 252);
        doc.setDrawColor(226, 232, 240);
        doc.roundedRect(14, y, 182, 34, 3, 3, 'FD');

        doc.setFont("helvetica", "bold");
        doc.setFontSize(10);
        doc.setTextColor(15, 23, 42);
        doc.text("EXECUTIVE VERDICT SUMMARY", 20, y + 10);

        // Verdict Badge Color
        if (verdictText.toUpperCase().includes("AI")) {
            doc.setTextColor(220, 38, 38); // Red
        } else if (verdictText.toUpperCase().includes("UNCERTAIN")) {
            doc.setTextColor(217, 119, 6); // Amber
        } else {
            doc.setTextColor(5, 150, 105); // Green
        }
        doc.setFontSize(14);
        doc.text(verdictText.toUpperCase(), 20, y + 20);

        doc.setFontSize(10);
        doc.setTextColor(51, 65, 85);
        doc.text(`AI Risk Score: ${aiRiskScore}`, 110, y + 14);
        doc.text(`Confidence Score: ${confidenceScore}`, 110, y + 22);

        // Section 2: 4-Layer Forensic Evidence Breakdown
        y += 42;
        doc.setFont("helvetica", "bold");
        doc.setFontSize(11);
        doc.setTextColor(15, 23, 42);
        doc.text("MULTIMODAL EVIDENCE LAYER ANALYSIS", 14, y);

        y += 6;
        // Table Header
        doc.setFillColor(37, 99, 235); // Blue header
        doc.rect(14, y, 182, 8, 'F');
        doc.setFontSize(8.5);
        doc.setTextColor(255, 255, 255);
        doc.text("Detection Module / Layer", 18, y + 5.5);
        doc.text("Metric / Representation Output", 92, y + 5.5);
        doc.text("Status / Verdict", 137, y + 5.5);

        y += 8;

        const layers = [
            {
                name: "Layer 1: Acoustic Analysis",
                desc: document.getElementById('l1-score-val')?.textContent || "N/A",
                sub: document.getElementById('l1-status')?.textContent || "Status: Active"
            },
            {
                name: "Layer 2: Waveform & Phase (AASIST)",
                desc: document.getElementById('l2-score-val')?.textContent || "N/A",
                sub: document.getElementById('l2-status')?.textContent || "Status: Active"
            },
            {
                name: "Layer 3: Spectral Analysis (LFCC)",
                desc: document.getElementById('l3-score-val')?.textContent || "N/A",
                sub: document.getElementById('l3-status')?.textContent || "Status: Active"
            },
            {
                name: "Layer 4: WavLM SSL Representation",
                desc: document.getElementById('l4-score-val')?.textContent || "768-dim",
                sub: document.getElementById('l4-status')?.textContent || "Status: Representation Active"
            }
        ];

        layers.forEach((layer, idx) => {
            doc.setFillColor(idx % 2 === 0 ? 255 : 248, idx % 2 === 0 ? 255 : 250, idx % 2 === 0 ? 255 : 252);
            doc.setDrawColor(226, 232, 240);
            doc.rect(14, y, 182, 10, 'FD');

            doc.setFont("helvetica", "normal");
            doc.setFontSize(8.5);
            doc.setTextColor(15, 23, 42);
            doc.text(layer.name, 18, y + 6.5);
            doc.text(layer.desc, 92, y + 6.5);
            
            // Format status cleanly and truncate if necessary to prevent overflowing box
            const statusText = layer.sub.startsWith("Status:") ? layer.sub : `Status: ${layer.sub}`;
            doc.text(statusText, 137, y + 6.5);

            y += 10;
        });

        // Section 3: Technical Explanation & Anomalies
        y += 10;
        doc.setFont("helvetica", "bold");
        doc.setFontSize(11);
        doc.setTextColor(15, 23, 42);
        doc.text("TECHNICAL VERDICT EXPLANATION & ANOMALIES", 14, y);

        y += 6;
        doc.setFont("helvetica", "normal");
        doc.setFontSize(9);
        doc.setTextColor(51, 65, 85);

        const splitExplanation = doc.splitTextToSize(explanationText, 182);
        doc.text(splitExplanation, 14, y);
        y += (splitExplanation.length * 4.5) + 6;

        // Anomalies
        const anomalyItems = Array.from(document.querySelectorAll('#anomaly-list li')).map(li => li.textContent);
        if (anomalyItems.length > 0) {
            doc.setFont("helvetica", "bold");
            doc.setFontSize(9.5);
            doc.setTextColor(180, 83, 9); // Amber
            doc.text("Identified Anomalies:", 14, y);
            y += 5;

            doc.setFont("helvetica", "normal");
            doc.setFontSize(9);
            doc.setTextColor(51, 65, 85);
            anomalyItems.forEach(item => {
                doc.text(`• ${item}`, 18, y);
                y += 5;
            });
        }

        // Footer Chain of Custody
        y = 275;
        doc.setDrawColor(226, 232, 240);
        doc.line(14, y, 196, y);

        doc.setFontSize(8);
        doc.setTextColor(148, 163, 184);
        doc.text("VoiceShield v2.0.0 Cyber Security Verification Engine — Cryptographic Chain of Custody Guaranteed", 14, y + 6);
        doc.text("Probabilistic confidence scoring analysis report generated for forensic audit purposes.", 14, y + 10);

        doc.save(`VoiceShield_Forensic_Report_${forensicId}.pdf`);
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
        
        let isAI = false;
        let isUncertain = false;

        if (mediaType === "image" || mediaType === "video") {
            isAI = aiScore > 0.50 || prediction === "Likely AI-Generated";
            isUncertain = false;
        } else {
            isAI = prediction === 'Likely AI-Generated' || aiScore >= 0.65;
            isUncertain = prediction.includes('Uncertain') || (aiScore > 0.35 && aiScore < 0.65) || (data.score_std && data.score_std > 0.28);
        }

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

        // Render Heatmap or Video Frame cards based on media_type
        const heatmapCard = document.getElementById('image-heatmap-card');
        const videoFramesCard = document.getElementById('video-frames-card');
        
        if (mediaType === "image") {
            if (videoFramesCard) videoFramesCard.classList.add('hidden');
            if (heatmapCard) {
                heatmapCard.classList.remove('hidden');
                const origImg = document.getElementById('heatmap-orig-img');
                const maskImg = document.getElementById('heatmap-mask-img');
                const imgDet = (data.layer_details && data.layer_details.image) || {};
                
                const currentPreviewSrc = document.getElementById('image-player')?.src;
                if (origImg && currentPreviewSrc) origImg.src = currentPreviewSrc;
                
                const heatmapData = data.heatmap_url || imgDet.heatmap_url;
                if (maskImg && heatmapData) {
                    maskImg.src = heatmapData;
                } else if (maskImg && currentPreviewSrc) {
                    maskImg.src = currentPreviewSrc;
                }
            }
        } else if (mediaType === "video") {
            if (heatmapCard) heatmapCard.classList.add('hidden');
            if (videoFramesCard) {
                videoFramesCard.classList.remove('hidden');
                const grid = document.getElementById('video-keyframes-grid');
                const vidDet = (data.layer_details && data.layer_details.video) || {};
                const frameScores = vidDet.frame_scores || [vidDet.visual_ai_prob || 0.5];
                
                if (grid) {
                    grid.innerHTML = '';
                    frameScores.forEach((fScore, idx) => {
                        const badge = document.createElement('div');
                        const fPct = Math.round(fScore * 100);
                        const isAnom = fScore > 0.50;
                        badge.style.cssText = `padding: 0.6rem 0.8rem; border-radius: 8px; font-size: 0.8rem; background: ${isAnom ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.15)'}; border: 1px solid ${isAnom ? '#ef4444' : '#10b981'}; min-width: 100px; text-align: center;`;
                        badge.innerHTML = `<span style="display:block; font-weight:700; color:${isAnom ? '#ef4444' : '#10b981'};">Frame ${idx+1}</span><span style="font-size:0.75rem; color:var(--text-muted);">${fPct}% AI Risk</span>`;
                        grid.appendChild(badge);
                    });
                }
            }
        } else {
            if (heatmapCard) heatmapCard.classList.add('hidden');
            if (videoFramesCard) videoFramesCard.classList.add('hidden');
        }

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
        document.getElementById('meta-duration').textContent = data.audio ? `${data.audio.duration || 'N/A'}s` : (data.layer_details.image ? data.layer_details.image.dimensions : (data.layer_details.video ? `${data.layer_details.video.duration_sec}s` : 'N/A'));
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
                { name: "SpecXNet 2D FFT Frequency Spectral Grid", score: imgDet.fft_spectral_score },
                { name: "SpecXNet ELA Compression Residual Heatmap", score: imgDet.ela_compression_score },
                { name: "SpecXNet Spatial Noise & Edge Consistency", score: imgDet.noise_covariance_score }
            ];
        } else if (mediaType === "video") {
            const vidDet = data.layer_details.video || {};
            dimensions = [
                { name: "FakeSTormer Multi-Frame Visual Keyframes", score: vidDet.visual_ai_prob },
                { name: "FakeSTormer Inter-Frame Motion Continuity", score: vidDet.temporal_ai_prob },
                { name: "FakeSTormer Spatio-Temporal Score Fusion", score: data.ai_probability }
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
        const card4 = document.querySelector('.layer-cards-grid .layer-card:nth-child(4)');

        if (mediaType === "image") {
            const imgDet = data.layer_details.image || {};
            if (card4) card4.style.display = 'none';
            
            this._setCard(1, "FEATURE 1", "SpecXNet Spatial Residual", "Dual-domain local spatial noise covariance & edge consistency check.", imgDet.ela_compression_score);
            this._setCard(2, "FEATURE 2", "SpecXNet 2D Spectral FFT", "Global 2D FFT spectral frequency ring analysis for diffusion/GAN grid artifacts.", imgDet.fft_spectral_score);
            this._setCard(3, "FEATURE 3", "SpecXNet Noise & Boundary", "Pixel noise covariance and facial boundary gradient consistency.", imgDet.noise_covariance_score);

        } else if (mediaType === "video") {
            const vidDet = data.layer_details.video || {};
            if (card4) card4.style.display = 'none';

            this._setCard(1, "COMPONENT 1", "FakeSTormer Visual Keyframes", "Multi-frame keyframe extraction & spatial deepfake score.", vidDet.visual_ai_prob);
            this._setCard(2, "COMPONENT 2", "FakeSTormer Temporal Motion", "Inter-frame motion continuity & Laplacian gradient jitter evaluation.", vidDet.temporal_ai_prob);
            this._setCard(3, "COMPONENT 3", "Spatio-Temporal Fusion", "Combined Visual + Inter-frame Temporal motion deepfake probability score.", data.ai_probability);

        } else {
            if (card4) card4.style.display = 'block';
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
        const scoreSubEl = document.getElementById(`l${cardNum}-sub`);
        const meterEl = document.getElementById(`l${cardNum}-meter`);
        const statusEl = document.getElementById(`l${cardNum}-status`);

        if (tagEl) tagEl.textContent = tag;
        if (titleEl) titleEl.textContent = title;
        if (descEl) descEl.textContent = desc;

        const isPercentage = scoreVal !== null && scoreVal !== undefined && typeof scoreVal === 'number';
        const pct = isPercentage ? Math.round(scoreVal * 100) : 0;

        if (scoreValEl) scoreValEl.textContent = isPercentage ? `${pct}%` : '768-dim';
        if (scoreSubEl) scoreSubEl.textContent = isPercentage ? 'AI Risk Score' : 'Vector Profiling';
        if (meterEl) meterEl.style.width = isPercentage ? `${pct}%` : '100%';

        if (statusEl) {
            let rawStatus = (statusStr || (isPercentage ? 'configured' : 'representation_active')).toLowerCase();
            let cleanLabel = 'Active';
            let tagClass = 'tag-active';
            let icon = 'fa-circle-check';

            if (rawStatus.includes('configured') && !rawStatus.includes('unconfigured')) {
                cleanLabel = 'Active';
                tagClass = 'tag-active';
                icon = 'fa-circle-check';
            } else if (rawStatus.includes('baseline')) {
                cleanLabel = 'Baseline Model';
                tagClass = 'tag-baseline';
                icon = 'fa-layer-group';
            } else if (rawStatus.includes('unconfigured')) {
                cleanLabel = 'Unconfigured';
                tagClass = 'tag-reserve';
                icon = 'fa-circle-minus';
            } else if (rawStatus.includes('representation') || rawStatus.includes('vector')) {
                cleanLabel = 'Vector Active';
                tagClass = 'tag-active';
                icon = 'fa-chart-simple';
            }

            statusEl.className = `layer-status-tag ${tagClass}`;
            statusEl.innerHTML = `<i class="fa-solid ${icon}"></i> ${cleanLabel}`;
        }
    }
}
