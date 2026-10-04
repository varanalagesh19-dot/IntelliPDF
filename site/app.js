/**
 * IntelliPDF — Interactive Landing Page & Live Embed Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const iframe = document.getElementById('streamlitFrame');
  const iframeLoader = document.getElementById('iframeLoader');
  const appUrlInput = document.getElementById('appUrlInput');
  const btnReloadIframe = document.getElementById('btnReloadIframe');
  const btnConfigureUrl = document.getElementById('btnConfigureUrl');
  const btnBannerConfig = document.getElementById('btnBannerConfig');
  const btnFullscreenEmbed = document.getElementById('btnFullscreenEmbed');
  const btnExternalLaunch = document.getElementById('btnExternalLaunch');
  const browserWindow = document.querySelector('.browser-window');

  // Modals
  const urlConfigDialog = document.getElementById('urlConfigDialog');
  const btnCloseConfigModal = document.getElementById('btnCloseConfigModal');
  const btnCancelConfig = document.getElementById('btnCancelConfig');
  const btnSaveConfig = document.getElementById('btnSaveConfig');
  const customUrlField = document.getElementById('customUrlField');
  const presetBtns = document.querySelectorAll('.preset-btn');

  const deployModal = document.getElementById('deployModal');
  const btnOpenDeployModal = document.getElementById('btnOpenDeployModal');
  const btnCloseDeployModal = document.getElementById('btnCloseDeployModal');
  const btnDismissDeployModal = document.getElementById('btnDismissDeployModal');

  // Mobile menu
  const mobileMenuBtn = document.getElementById('mobileMenuBtn');
  const navMenu = document.getElementById('navMenu');

  // Mark-wise interactive demo elements
  const markButtons = document.querySelectorAll('.mark-btn');
  const demoOutputStatus = document.getElementById('demoOutputStatus');
  const demoOutputText = document.getElementById('demoOutputText');

  // Default app URL handling (with localStorage persistence)
  const STORAGE_KEY = 'intellipdf_streamlit_url';
  const DEFAULT_URL = 'https://intellipdf.streamlit.app';

  function getStoredUrl() {
    return localStorage.getItem(STORAGE_KEY) || DEFAULT_URL;
  }

  function setAppUrl(url) {
    let cleanUrl = url.trim();
    if (!cleanUrl.startsWith('http://') && !cleanUrl.startsWith('https://')) {
      cleanUrl = 'https://' + cleanUrl;
    }
    localStorage.setItem(STORAGE_KEY, cleanUrl);
    
    // Format for embed
    const embedUrl = cleanUrl.includes('?') ? `${cleanUrl}&embed=true` : `${cleanUrl}?embed=true`;
    
    appUrlInput.value = cleanUrl;
    btnExternalLaunch.href = cleanUrl;
    
    // Show loading spinner
    iframeLoader.classList.remove('hidden');
    iframe.src = embedUrl;
  }

  // Initial load
  const initialUrl = getStoredUrl();
  setAppUrl(initialUrl);

  // When iframe finishes loading, hide spinner
  iframe.addEventListener('load', () => {
    iframeLoader.classList.add('hidden');
  });

  // Reload iframe
  btnReloadIframe.addEventListener('click', () => {
    iframeLoader.classList.remove('hidden');
    iframe.src = iframe.src;
  });

  // Direct URL input change
  appUrlInput.addEventListener('change', () => {
    setAppUrl(appUrlInput.value);
  });
  appUrlInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      setAppUrl(appUrlInput.value);
    }
  });

  // Fullscreen Embed Toggle
  btnFullscreenEmbed.addEventListener('click', () => {
    browserWindow.classList.toggle('fullscreen');
    if (browserWindow.classList.contains('fullscreen')) {
      btnFullscreenEmbed.innerHTML = `
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="4 14 10 14 10 20"></polyline><polyline points="20 10 14 10 14 4"></polyline><line x1="14" y1="10" x2="21" y2="3"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>
        <span>Exit</span>
      `;
    } else {
      btnFullscreenEmbed.innerHTML = `
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>
        <span>Fullscreen</span>
      `;
    }
  });

  // Dialog Controls (using HTML5 native <dialog>)
  function openUrlModal() {
    customUrlField.value = getStoredUrl();
    if (typeof urlConfigDialog.showModal === 'function') {
      urlConfigDialog.showModal();
    } else {
      urlConfigDialog.setAttribute('open', '');
    }
  }

  btnConfigureUrl.addEventListener('click', openUrlModal);
  if (btnBannerConfig) {
    btnBannerConfig.addEventListener('click', openUrlModal);
  }

  function closeUrlModal() {
    if (typeof urlConfigDialog.close === 'function') {
      urlConfigDialog.close();
    } else {
      urlConfigDialog.removeAttribute('open');
    }
  }

  btnCloseConfigModal.addEventListener('click', closeUrlModal);
  btnCancelConfig.addEventListener('click', closeUrlModal);

  btnSaveConfig.addEventListener('click', () => {
    if (customUrlField.value) {
      setAppUrl(customUrlField.value);
    }
    closeUrlModal();
  });

  presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      customUrlField.value = btn.dataset.url;
    });
  });

  // Deploy Info Modal
  btnOpenDeployModal.addEventListener('click', () => {
    if (typeof deployModal.showModal === 'function') {
      deployModal.showModal();
    } else {
      deployModal.setAttribute('open', '');
    }
  });

  function closeDeployModal() {
    if (typeof deployModal.close === 'function') {
      deployModal.close();
    } else {
      deployModal.removeAttribute('open');
    }
  }

  btnCloseDeployModal.addEventListener('click', closeDeployModal);
  btnDismissDeployModal.addEventListener('click', closeDeployModal);

  // Close modals on outside backdrop click
  [urlConfigDialog, deployModal].forEach(dialog => {
    dialog.addEventListener('click', (e) => {
      const rect = dialog.getBoundingClientRect();
      const isInDialog = (
        rect.top <= e.clientY &&
        e.clientY <= rect.top + rect.height &&
        rect.left <= e.clientX &&
        e.clientX <= rect.left + rect.width
      );
      if (!isInDialog) {
        if (typeof dialog.close === 'function') dialog.close();
      }
    });
  });

  // Mobile menu toggle
  mobileMenuBtn.addEventListener('click', () => {
    const isExpanded = navMenu.style.display === 'flex';
    navMenu.style.display = isExpanded ? 'none' : 'flex';
    if (!isExpanded) {
      navMenu.style.position = 'absolute';
      navMenu.style.top = '100%';
      navMenu.style.left = '0';
      navMenu.style.right = '0';
      navMenu.style.flexDirection = 'column';
      navMenu.style.background = 'rgba(7, 9, 14, 0.95)';
      navMenu.style.padding = '20px';
      navMenu.style.borderBottom = '1px solid rgba(255, 255, 255, 0.1)';
    }
  });

  // -------------------------------------------------------------
  // Interactive Mark-Wise Answer Demo Synthesizer
  // -------------------------------------------------------------
  const markResponses = {
    2: {
      status: "Displaying 2-Mark Standard Answer (Crisp Definition)",
      html: `
        <p><strong>Definition:</strong> A Convolutional Neural Network (CNN) is a specialized class of deep feed-forward artificial neural networks designed primarily for processing structured grid data such as images.</p>
        <p><strong>Core Mechanism:</strong> It uses mathematical convolution operations instead of standard matrix multiplication in at least one layer to automatically learn spatial hierarchies of features.</p>
        <div class="citation-chip">Page 42, Paragraph 2</div>
      `
    },
    5: {
      status: "Displaying 5-Mark Conceptual Answer (Key Components)",
      html: `
        <p><strong>Convolutional Neural Networks (CNNs)</strong> process visual imagery by exploiting local spatial correlations through shared filter weights.</p>
        <p><strong>Key Architectural Layers:</strong></p>
        <ul>
          <li><strong>Convolutional Layer:</strong> Convolves learnable filters (kernels) across input tensors to create feature maps capturing edges, textures, and patterns.</li>
          <li><strong>Activation Function (ReLU):</strong> Introduces non-linearity: <code>f(x) = max(0, x)</code>, eliminating vanishing gradient risks during backpropagation.</li>
          <li><strong>Pooling Layer (Max/Average):</strong> Downsamples spatial dimensions (height & width), reducing computational complexity and enforcing translation invariance.</li>
          <li><strong>Fully Connected (Dense) Layer:</strong> Flattens multidimensional features into a 1D vector to produce final classification logits.</li>
        </ul>
        <div class="citation-chip">Page 43, Section 3.2</div>
      `
    },
    16: {
      status: "Displaying 16-Mark Comprehensive Essay (University Exam Format)",
      html: `
        <p><strong>1. Introduction & Background:</strong><br>
        Convolutional Neural Networks (CNNs) were introduced by Yann LeCun (LeNet-5) to overcome the limitation of Multi-Layer Perceptrons (MLPs) regarding parameter explosion when processing high-resolution images.</p>
        
        <p><strong>2. Mathematical Formulation of Convolution:</strong><br>
        For an input image \(I\) and kernel \(K\) of dimension \((2m+1) \times (2n+1)\):<br>
        <code>S(i, j) = (I * K)(i, j) = ∑_m ∑_n I(i - m, j - n) K(m, n)</code></p>

        <p><strong>3. Deep Layer-by-Layer Walkthrough:</strong></p>
        <ul>
          <li><strong>Feature Extraction Stage:</strong> Alternating Conv2D and MaxPool2D blocks hierarchically extract low-level edges up to high-level semantic object detectors.</li>
          <li><strong>Regularization:</strong> Spatial Dropout and Batch Normalization accelerate convergence and suppress overfitting.</li>
          <li><strong>Softmax Logit Classification:</strong> The flattened representation maps to discrete class probabilities: <code>P(Y=k|x) = exp(z_k) / ∑ exp(z_j)</code>.</li>
        </ul>

        <p><strong>4. Key Strengths vs Traditional ML:</strong> Parameter sharing (sparse connectivity) reduces weights from \(O(N \times M)\) to fixed kernel sizes \(K \times K\), granting invariant feature recognition.</p>
        
        <div class="citation-chip">Page 42-45, Complete Chapter 4</div>
      `
    }
  };

  markButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      markButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      const marks = btn.dataset.marks;
      if (markResponses[marks]) {
        demoOutputStatus.textContent = markResponses[marks].status;
        demoOutputText.innerHTML = markResponses[marks].html;
      }
    });
  });
});
