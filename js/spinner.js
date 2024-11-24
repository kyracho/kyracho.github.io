// funtion for the loading spinner
window.addEventListener('load', () => {
    setTimeout(() => {
      document.getElementById('loading-icon').style.display = 'none';
      document.getElementById('iframe-content').style.display = 'block';
    }, 5000);
  });