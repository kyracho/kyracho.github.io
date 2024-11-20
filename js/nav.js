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
    rootMargin: "0px 0px -50% 0px", // Trigger when section crosses 50% of the viewport
    threshold: 0, // Trigger as soon as a section enters the threshold
  });

  // Observe all sections
  sections.forEach((section) => observer.observe(section));

  // Click event for dots to navigate to sections
  dots.forEach((dot) => {
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
