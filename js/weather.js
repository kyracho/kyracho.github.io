async function fetchWeather() {
    const apiKey = '209876685b5040a389a73100241909'; // Replace with your actual API key
    const city = 'New York'; // Replace with the city you want to fetch the weather for
    const url = `https://api.weatherapi.com/v1/current.json?key=${apiKey}&q=${city}&aqi=no`;
  
    try {
      const response = await fetch(url);
      const weatherData = await response.json();
      displayWeather(weatherData);
    } catch (error) {
      console.error('Error fetching weather data:', error);
    }
  }
  
  function displayWeather(data) {
    const weatherElement = document.getElementById('weather');
    const temperature = data.current.temp_c; // Temperature in Celsius
    const condition = data.current.condition.text; // Weather condition (e.g., Sunny, Rainy)
    const icon = data.current.condition.icon; // Icon URL
  
    // weatherElement.innerHTML = `
    //   <img src="${icon}" alt="Weather Icon">
    //   <p>Temperature: ${temperature}°C</p>
    //   <p>Condition: ${condition}</p>
    // `;
    weatherElement.innerHTML = `
      <p style="font-size:13px;">Temperature: ${temperature}°C</p>
    `;
  }
  
  fetchWeather();
  