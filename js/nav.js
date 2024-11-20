document.addEventListener("DOMContentLoaded", () => {
    const scroller = document.querySelector(".scroller");
    const sections = scroller.querySelectorAll("section");
    const dots = document.querySelectorAll(".dot");
  
    function updateActiveDot() {
      let currentSection = "";
  
      sections.forEach(section => {
        const sectionTop = section.getBoundingClientRect().top;
      
        if (sectionTop >= 0 && sectionTop < window.innerHeight / 2) {
          currentSection = section.getAttribute("id");
        }
      });
      
  
      dots.forEach(dot => {
        dot.classList.toggle("active", dot.dataset.section === currentSection);
      });
    }
  
    // Scroll listener on .scroller
    scroller.addEventListener("scroll", () => {
        updateActiveDot();
    });
  
    // Click event for dots to navigate to sections
    dots.forEach(dot => {
      dot.addEventListener("click", (event) => {
        event.preventDefault();
        const targetSection = document.getElementById(dot.dataset.section);
  
        // Scroll to target section within .scroller
        scroller.scrollTo({
          top: targetSection.offsetTop,
          behavior: "smooth",
        });
      });
    });
  });
  