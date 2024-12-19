document.addEventListener("DOMContentLoaded", () => {
  const scroller = document.querySelector(".scroller");
  const sections = document.querySelectorAll(".scroller section");
  const dots = document.querySelectorAll(".dot");

  // Observer callback to update active dot
  function handleIntersect(entries) {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const currentSection = entry.target.id;

        // Update the active dot
        dots.forEach((dot) => {
          dot.classList.toggle("active", dot.dataset.section === currentSection);
        });
      }
    });
  }

  // Create IntersectionObserver
  const observer = new IntersectionObserver(handleIntersect, {
    root: scroller, // Observe within the .scroller container
    rootMargin: "0px", // No additional margins
    threshold: 0.5, // Trigger when section is 50% visible
  });

  // Observe all sections
  sections.forEach((section) => observer.observe(section));

  // Click event for dots to navigate to sections
  dots.forEach((dot) => {
    dot.addEventListener("click", (event) => {
      event.preventDefault();
      const targetSection = document.getElementById(dot.dataset.section);

      // Scroll to target section within .scroller (horizontal scrolling)
      scroller.scrollTo({
        left: targetSection.offsetLeft, // Use offsetLeft for horizontal scroll
        behavior: "smooth",
      });
    });
  });
});
