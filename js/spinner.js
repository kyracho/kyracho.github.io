// funtion for the loading spinner
// Replace spinner with iframe content once iframe is fully loaded
const iframe = document.getElementById('iframe-content');
  iframe.onload = () => {
    document.getElementById('loading-icon').style.display = 'none';
    iframe.style.display = 'block';
  };
