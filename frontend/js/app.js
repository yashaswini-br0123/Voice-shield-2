document.addEventListener('DOMContentLoaded', () => {
    // Dynamic Base API URL handling (works both on http://localhost:8000 AND file:// origins)
    const API_BASE_URL = window.location.protocol.startsWith('http') ? '' : 'http://localhost:8000';

    const visualizer = new AudioVisualizer('visualizer-canvas');
    const dashboard = new DashboardManager();
    let currentSelectedFile = null;

    // Sidebar Tab Buttons
    const tabAll = document.getElementById('tab-all');
    const tabAudio = document.getElementById('tab-audio');
    const tabImage = document.getElementById('tab-image');
    const tabVideo = document.getElementById('tab-video');
    const tabRecord = document.getElementById('tab-record');

    // Views
    const viewUpload = document.getElementById('view-upload');
    const viewRecord = document.getElementById('view-record');

    // Elements
    const dropzone = document.getElementById('dropzone');
    const dropSubtitle = document.getElementById('dropzone-subtitle');
    const fileInput = document.getElementById('media-file-input');
    const previewCard = document.getElementById('media-preview-card');
    const previewFilename = document.getElementById('preview-filename');
    const previewMeta = document.getElementById('preview-meta');
    const previewIcon = document.getElementById('preview-type-icon');

    // Players
    const audioContainer = document.getElementById('audio-preview-container');
    const imageContainer = document.getElementById('image-preview-container');
    const videoContainer = document.getElementById('video-preview-container');
    
    const audioPlayer = document.getElementById('audio-player');
    const imagePlayer = document.getElementById('image-player');
    const videoPlayer = document.getElementById('video-player');

    const btnClear = document.getElementById('btn-clear-media');
    const btnAnalyze = document.getElementById('btn-analyze');
    const btnRetry = document.getElementById('btn-retry-analysis');
    
    const quickVoiceGroup = document.getElementById('quick-voice-group');
    const quickImageGroup = document.getElementById('quick-image-group');
    const quickVideoGroup = document.getElementById('quick-video-group');

    const btnTestHuman = document.getElementById('btn-test-human');
    const btnTestAI = document.getElementById('btn-test-ai');
    const btnTestRealImg = document.getElementById('btn-test-real-img');
    const btnTestAIImg = document.getElementById('btn-test-ai-img');
    const btnTestRealVideo = document.getElementById('btn-test-real-video');
    const btnTestAIVideo = document.getElementById('btn-test-ai-video');

    const backendPill = document.getElementById('backend-status');
    const demoPill = document.getElementById('demo-status');

    // 0. Cybersecurity Intro Splash Auto-Dismiss
    const splash = document.getElementById('intro-splash');
    if (splash) {
        setTimeout(() => {
            splash.classList.add('fade-out');
            setTimeout(() => splash.remove(), 500);
        }, 1800);
    }

    // 1. Health Check
    checkBackendHealth();

    async function checkBackendHealth() {
        try {
            let res = await fetch(`${API_BASE_URL}/health`);
            if (!res.ok) {
                res = await fetch(`${API_BASE_URL}/api/health`);
            }
            if (res.ok) {
                const data = await res.json();
                backendPill.className = 'status-pill status-online';
                backendPill.innerHTML = '<span class="dot"></span> Backend: Online';

                if (data.demo_mode) {
                    demoPill.classList.remove('hidden');
                } else {
                    demoPill.classList.add('hidden');
                }
            } else {
                throw new Error("API health check non-200");
            }
        } catch (err) {
            backendPill.className = 'status-pill status-loading';
            backendPill.innerHTML = '<span class="dot"></span> Backend: Offline / Connecting...';
        }
    }


    // 2. Sidebar Navigation Switcher
    const allTabs = [tabAll, tabAudio, tabImage, tabVideo, tabRecord];
    function activateTab(activeTabBtn) {
        allTabs.forEach(btn => btn?.classList.remove('active'));
        activeTabBtn.classList.add('active');

        if (activeTabBtn === tabRecord) {
            viewUpload.classList.add('hidden');
            viewRecord.classList.remove('hidden');
        } else {
            viewUpload.classList.remove('hidden');
            viewRecord.classList.add('hidden');
        }
    }

    function showAllQuickDemoGroups() {
        quickVoiceGroup?.classList.remove('hidden');
        quickImageGroup?.classList.remove('hidden');
        quickVideoGroup?.classList.remove('hidden');
    }

    tabAll?.addEventListener('click', () => {
        activateTab(tabAll);
        fileInput.accept = "audio/*,video/*,image/*";
        if (dropSubtitle) dropSubtitle.textContent = "Supports ALL Audio Formats, Images & Videos";
        showAllQuickDemoGroups();
    });

    tabAudio?.addEventListener('click', () => {
        activateTab(tabAudio);
        fileInput.accept = "audio/*";
        if (dropSubtitle) dropSubtitle.textContent = "Supports ALL Audio Formats (WAV, MP3, AAC, M4A, OPUS)";
        showAllQuickDemoGroups();
    });

    tabImage?.addEventListener('click', () => {
        activateTab(tabImage);
        fileInput.accept = "image/*";
        if (dropSubtitle) dropSubtitle.textContent = "Supports Images (JPG, PNG, WEBP, BMP)";
        showAllQuickDemoGroups();
    });

    tabVideo?.addEventListener('click', () => {
        activateTab(tabVideo);
        fileInput.accept = "video/*";
        if (dropSubtitle) dropSubtitle.textContent = "Supports Deepfake Videos (MP4, AVI, MOV, MKV)";
        showAllQuickDemoGroups();
    });

    tabRecord?.addEventListener('click', () => {
        activateTab(tabRecord);
    });

    // 3. Quick Demo Test Triggers
    btnTestHuman?.addEventListener('click', () => createSyntheticSampleWav(120.0, "human_voice_sample.wav"));
    btnTestAI?.addEventListener('click', () => createSyntheticSampleWav(140.0, "ai_synthetic_voice_sample.wav", true));
    
    btnTestRealImg?.addEventListener('click', () => createSyntheticImageFile("real_photo_sample.jpg", false));
    btnTestAIImg?.addEventListener('click', () => createSyntheticImageFile("ai_generated_image.jpg", true));

    btnTestRealVideo?.addEventListener('click', () => createSyntheticVideoFile("real_video_sample.webm", false));
    btnTestAIVideo?.addEventListener('click', () => createSyntheticVideoFile("ai_deepfake_video.webm", true));

    function createSyntheticImageFile(filename, isAI = false) {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(512, 512);
        const data = imgData.data;

        for (let y = 0; y < 512; y++) {
            for (let x = 0; x < 512; x++) {
                const idx = (y * 512 + x) * 4;
                if (!isAI) {
                    const r = Math.sin(x * 0.01) * 70 + 120 + (Math.random() - 0.5) * 20;
                    const g = Math.cos(y * 0.01) * 70 + 120 + (Math.random() - 0.5) * 20;
                    const b = Math.sin((x + y) * 0.008) * 70 + 140 + (Math.random() - 0.5) * 20;
                    data[idx] = Math.max(0, Math.min(255, r));
                    data[idx + 1] = Math.max(0, Math.min(255, g));
                    data[idx + 2] = Math.max(0, Math.min(255, b));
                    data[idx + 3] = 255;
                } else {
                    const gridPattern = (Math.floor(x / 4) + Math.floor(y / 4)) % 2 === 0 ? 55 : -55;
                    const baseR = 100 + Math.sin(x * 0.05) * 50;
                    const baseG = 120 + Math.cos(y * 0.05) * 50;
                    const baseB = 200 + Math.sin((x - y) * 0.05) * 50;
                    data[idx] = Math.max(0, Math.min(255, baseR + gridPattern));
                    data[idx + 1] = Math.max(0, Math.min(255, baseG + gridPattern));
                    data[idx + 2] = Math.max(0, Math.min(255, baseB + gridPattern));
                    data[idx + 3] = 255;
                }
            }
        }
        ctx.putImageData(imgData, 0, 0);

        canvas.toBlob((blob) => {
            const testImgFile = new File([blob], filename, { type: 'image/jpeg' });
            setMediaFile(testImgFile, "image", blob, true);
        }, 'image/jpeg', 0.9);
    }

    function createSyntheticVideoFile(filename, isAI = false) {
        const canvas = document.createElement('canvas');
        canvas.width = 320;
        canvas.height = 240;
        const ctx = canvas.getContext('2d');
        
        const stream = canvas.captureStream(30);
        
        let audioCtx = null;
        try {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const dest = audioCtx.createMediaStreamDestination();
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.frequency.value = isAI ? 140 : 120;
            osc.connect(gain);
            gain.connect(dest);
            osc.start();
            dest.stream.getAudioTracks().forEach(track => stream.addTrack(track));
        } catch (e) {}

        let frameCount = 0;
        const interval = setInterval(() => {
            frameCount++;
            if (!isAI) {
                ctx.fillStyle = '#0f172a';
                ctx.fillRect(0, 0, 320, 240);
                const r = 100 + Math.sin(frameCount * 0.1) * 40;
                ctx.fillStyle = `rgb(${r}, 150, 200)`;
                ctx.beginPath();
                ctx.arc(160, 120, 40, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#ffffff';
                ctx.font = '14px sans-serif';
                ctx.fillText('AUTHENTIC REAL VIDEO SAMPLE', 30, 40);
            } else {
                ctx.fillStyle = '#1e1b4b';
                ctx.fillRect(0, 0, 320, 240);
                ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
                for (let i = 0; i < 320; i += 8) {
                    for (let j = 0; j < 240; j += 8) {
                        if ((i + j) % 16 === 0) ctx.fillRect(i, j, 4, 4);
                    }
                }
                ctx.fillStyle = '#ffffff';
                ctx.font = '14px sans-serif';
                ctx.fillText('SYNTHETIC AI VIDEO SAMPLE', 30, 40);
            }
        }, 33);

        let mimeType = 'video/webm';
        if (!MediaRecorder.isTypeSupported('video/webm')) {
            mimeType = 'video/mp4';
        }

        const rec = new MediaRecorder(stream, { mimeType });
        const chunks = [];
        rec.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
        rec.onstop = () => {
            clearInterval(interval);
            if (audioCtx) audioCtx.close();
            const blob = new Blob(chunks, { type: mimeType });
            const testVidFile = new File([blob], filename, { type: mimeType });
            setMediaFile(testVidFile, "video", blob, true);
        };

        rec.start();
        setTimeout(() => rec.stop(), 2000);
    }

    function createSyntheticSampleWav(f0_base, filename, isAI = false) {
        const sr = 16000;
        const duration = 3.0;
        const numSamples = sr * duration;
        const samples = new Int16Array(numSamples);

        for (let i = 0; i < numSamples; i++) {
            const t = i / sr;
            let sampleVal = 0;

            if (!isAI) {
                // REAL HUMAN VOICE SPEECH SIMULATION:
                // 1. Natural Fundamental Frequency (F0) with pitch vibrato (~5.5 Hz modulation, 6 Hz vibrato depth)
                const f0 = f0_base + 6.0 * Math.sin(2 * Math.PI * 5.5 * t) + 2.0 * Math.sin(2 * Math.PI * 1.8 * t);
                
                // 2. Vocal Tract Formant Resonances (F1 ~ 500 Hz, F2 ~ 1500 Hz, F3 ~ 2500 Hz)
                const phase0 = 2 * Math.PI * f0 * t;
                const phase1 = 2 * Math.PI * 500 * t;
                const phase2 = 2 * Math.PI * 1500 * t;
                const phase3 = 2 * Math.PI * 2500 * t;
                
                // Harmonic glottal pulse decay (1/n) + formant shaping
                const glottal = Math.sin(phase0) + 0.5 * Math.sin(2 * phase0) + 0.25 * Math.sin(3 * phase0);
                const formants = 0.4 * Math.sin(phase1) + 0.25 * Math.sin(phase2) + 0.15 * Math.sin(phase3);
                
                // 3. Speaking cadence envelope (syllables ~3.5 Hz)
                const envelope = 0.5 + 0.5 * Math.sin(2 * Math.PI * 3.5 * t);
                
                // 4. Natural micro-jitter and room acoustic noise floor
                const jitterNoise = (Math.random() - 0.5) * 0.08;
                
                sampleVal = (glottal * 0.4 + formants * 0.5) * envelope + jitterNoise;
            } else {
                // SYNTHETIC AI VOICE SIMULATION:
                // 1. Rigid, static fundamental frequency (zero pitch vibrato, jitter < 0.0001)
                const f0 = f0_base;
                const phase = 2 * Math.PI * f0 * t;
                
                // 2. Pure un-filtered tone without vocal tract formants
                const tone = Math.sin(phase) + 0.3 * Math.sin(2 * phase);
                
                // 3. Constant robotic amplitude without speech cadence envelope
                sampleVal = tone * 0.7;
            }

            samples[i] = Math.max(-32768, Math.min(32767, sampleVal * 14000));
        }

        const buffer = new ArrayBuffer(44 + numSamples * 2);
        const view = new DataView(buffer);

        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + numSamples * 2, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sr, true);
        view.setUint32(28, sr * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(view, 36, 'data');
        view.setUint32(40, numSamples * 2, true);

        for (let i = 0; i < numSamples; i++) {
            view.setInt16(44 + i * 2, samples[i], true);
        }

        const blob = new Blob([view], { type: 'audio/wav' });
        const testFile = new File([blob], filename, { type: 'audio/wav' });

        setMediaFile(testFile, "audio", blob, true);
    }

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    // 4. Microphone Recorder Setup
    const recorder = new VoiceRecorder(visualizer, (file, blob) => {
        setMediaFile(file, "audio", blob, true);
    });

    // 5. Clickable Sidebar Dropzone
    dropzone.addEventListener('click', (e) => {
        if (e.target.tagName !== 'BUTTON' && e.target.tagName !== 'INPUT') {
            fileInput.click();
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    function handleSelectedFile(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        const imageExts = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'];
        const videoExts = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'm4v'];

        let mediaCategory = "audio";
        if (imageExts.includes(ext) || file.type.startsWith('image/')) {
            mediaCategory = "image";
        } else if (videoExts.includes(ext) || file.type.startsWith('video/')) {
            mediaCategory = "video";
        } else {
            mediaCategory = "audio";
        }

        if (file.size > 50 * 1024 * 1024) {
            alert('File exceeds maximum size limit of 50MB.');
            return;
        }

        setMediaFile(file, mediaCategory, null, true);
    }

    function setMediaFile(file, category = "audio", blobOverride = null, autoRun = false) {
        currentSelectedFile = file;
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        
        previewFilename.textContent = file.name;
        previewMeta.textContent = `${sizeMB} MB • ${category.toUpperCase()}`;
        
        const fileUrl = URL.createObjectURL(blobOverride || file);

        audioContainer.classList.add('hidden');
        imageContainer.classList.add('hidden');
        videoContainer.classList.add('hidden');

        if (category === "image") {
            previewIcon.className = "fa-solid fa-file-image file-icon";
            imagePlayer.src = fileUrl;
            imageContainer.classList.remove('hidden');

        } else if (category === "video") {
            previewIcon.className = "fa-solid fa-file-video file-icon";
            videoPlayer.src = fileUrl;
            videoContainer.classList.remove('hidden');

        } else {
            previewIcon.className = "fa-solid fa-file-audio file-icon";
            audioPlayer.src = fileUrl;
            audioContainer.classList.remove('hidden');
        }

        previewCard.classList.remove('hidden');

        if (autoRun) {
            runAnalysisExecution();
        }
    }

    btnClear.addEventListener('click', () => {
        currentSelectedFile = null;
        fileInput.value = '';
        audioPlayer.src = '';
        imagePlayer.src = '';
        videoPlayer.src = '';
        previewCard.classList.add('hidden');
    });

    // 6. Analyze Media Execution
    btnAnalyze?.addEventListener('click', () => runAnalysisExecution());
    btnRetry?.addEventListener('click', () => runAnalysisExecution());

    async function runAnalysisExecution() {
        if (!currentSelectedFile) {
            alert('Please select, record, or click a demo sample first.');
            return;
        }

        btnAnalyze.disabled = true;
        btnAnalyze.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> ANALYZING DEEPFAKE...';

        try {
            const ext = currentSelectedFile.name.split('.').pop().toLowerCase();
            let detectedType = "audio";
            if (['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'].includes(ext) || currentSelectedFile.type.startsWith('image/')) detectedType = "image";
            if (['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'].includes(ext) || currentSelectedFile.type.startsWith('video/')) detectedType = "video";

            const stepperPromise = dashboard.runProgressStepper(detectedType);

            const formData = new FormData();
            formData.append('file', currentSelectedFile);

            const apiPromise = (async () => {
                try {
                    let res = await fetch(`${API_BASE_URL}/analyze`, { method: 'POST', body: formData });
                    if (!res.ok && res.status === 404) {
                        res = await fetch(`${API_BASE_URL}/api/analyze`, { method: 'POST', body: formData });
                    }
                    return res;
                } catch (e) {
                    await new Promise(r => setTimeout(r, 200));
                    return await fetch(`${API_BASE_URL}/analyze`, { method: 'POST', body: formData });
                }
            })();


            const [_, response] = await Promise.all([stepperPromise, apiPromise]);

            if (!response.ok) {
                const errJson = await response.json();
                throw new Error(errJson.detail || 'Media analysis failed.');
            }

            const data = await response.json();
            dashboard.renderResults(data);

        } catch (err) {
            checkBackendHealth();
            const errMsg = (err.message && err.message.includes('Failed to fetch'))
                ? 'Backend Server Unreachable (Failed to fetch). Please ensure the backend server is running on http://localhost:8000.'
                : (err.message || 'Media analysis failed.');
            dashboard.showError(errMsg);
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.innerHTML = '<i class="fa-solid fa-brain"></i> ANALYZE DEEPFAKE';
        }
    }
});
