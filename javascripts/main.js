document.addEventListener("DOMContentLoaded", async function () {
    let copyIcon = "";
    let copiedIcon = "";

    try {
        const [copyResponse, copiedResponse] = await Promise.all([
            fetch("images/copy.svg"),
            fetch("images/copied.svg"),
        ]);
        copyIcon = await copyResponse.text();
        copiedIcon = await copiedResponse.text();
    } catch (err) {
        console.error("Failed to load icons:", err);
        // Fallback text or icons
        copyIcon = "Copy";
        copiedIcon = "Copied!";
    }

    const codeElements = document.querySelectorAll("code");

    codeElements.forEach(function (codeElement) {
        const parent = codeElement.parentElement;
        let codeBlock = null;

        if (parent.tagName === "PRE") {
            codeBlock = parent;
        } else {
            const wrapper = document.createElement("div");
            wrapper.className = "code-container";
            wrapper.style.display = "block";
            wrapper.style.backgroundColor = "#fff";
            wrapper.style.padding = "10px";
            wrapper.style.borderRadius = "2px";
            wrapper.style.boxShadow = "0 0 10px rgba(0,0,0,.1)";
            wrapper.style.margin = "10px 0 15px 0";

            parent.insertBefore(wrapper, codeElement);
            wrapper.appendChild(codeElement);
            codeBlock = wrapper;
        }

        // Create a new wrapper for the code block and the button
        const outerWrapper = document.createElement("div");
        outerWrapper.className = "code-block-wrapper";
        codeBlock.parentNode.insertBefore(outerWrapper, codeBlock);
        outerWrapper.appendChild(codeBlock);

        const copyButton = document.createElement("button");
        copyButton.className = "copy-btn";
        copyButton.innerHTML = copyIcon;
        copyButton.setAttribute("data-tooltip", "Copy");

        copyButton.addEventListener("click", function () {
            const codeToCopy = codeElement.innerText;
            navigator.clipboard.writeText(codeToCopy).then(
                function () {
                    copyButton.innerHTML = copiedIcon;
                    copyButton.setAttribute("data-tooltip", "Copied!");
                    setTimeout(function () {
                        copyButton.innerHTML = copyIcon;
                        copyButton.setAttribute("data-tooltip", "Copy");
                    }, 2000);
                },
                function (err) {
                    console.error("Could not copy text: ", err);
                }
            );
        });

        // Append the button to the outer wrapper, not the code block
        outerWrapper.appendChild(copyButton);
    });
});
