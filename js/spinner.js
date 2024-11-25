// funtion for the loading spinner
// Replace spinner with iframe content once iframe is fully loaded, and once DOM is loaded. 
document.addEventListener('DOMContentLoaded', () => {
  const iframe = document.getElementById('iframe-content');
  if (iframe) {
    iframe.onload = () => {
      document.getElementById('loading-icon').style.display = 'none';
      iframe.style.display = 'block';
    };
  } else {
    console.error('Iframe not found!');
  }
});
