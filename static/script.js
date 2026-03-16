function setActive(element) {

    // remove active from all
    let links = document.querySelectorAll("nav a");
    links.forEach(link => link.classList.remove("active"));

    // add active to clicked
    element.classList.add("active");
}
