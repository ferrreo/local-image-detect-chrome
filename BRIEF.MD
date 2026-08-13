local AI challenge: AI image detector for Chrome

First submission to clear the bar takes home the bounty.

(This bounty is open-ended. Anyone may contribute additional funding to the prize pool at any time before a claim is submitted for vote. In return, you will receive voting rights on approving claims. Learn more here.)

the challenge
Build a Google Chrome extension that reliably detects AI-generated images using only the compute available inside the browser.

No cloud inference.
No external APIs.
No local server dependencies.
Everything must run entirely inside the browser runtime.
Install the extension, visit any webpage, and AI-generated images should automatically be identified with a confidence score.

The goal is to create a detector that is genuinely useful for everyday browsing while preserving user privacy by never sending image data to a remote server.

requirements
Your submission must:

Be fully open source under the MIT License.
Run as a native Manifest V3 Google Chrome extension.
Perform all inference locally using browser technologies such as WebGPU, WebAssembly, or WebGL.
The extension may perform a one-time download of publicly available model weights during initial setup. After that, all inference must run entirely offline and the extension may not download additional models, weights, or inference-related assets.
Automatically analyze images displayed on ordinary webpages.
Display a confidence score for every analyzed image.
Include complete build and installation instructions.
Be fully reproducible from source.
Any browser-local detection approach is permitted, including learned models, metadata analysis, watermark detection, hybrid pipelines, or other techniques.

the bar
To win this bounty, your extension must achieve at least 75.0% balanced accuracy on our evaluation benchmark while using a 65% confidence threshold.

Balanced accuracy gives equal weight to correctly identifying both real and AI-generated images.

The benchmark consists of a held-out set of real and AI-generated images assembled from publicly available datasets and additional web-realistic samples.

The exact evaluation images will remain private until the bounty concludes to discourage benchmark overfitting. The evaluation methodology and scoring process will remain consistent for every submission.

We evaluated an existing open-source browser-based detector—which itself relies on a local backend server rather than pure in-browser inference—and it failed to achieve 60% balanced accuracy on our benchmark. This suggests there is significant room for improvement under these constraints.

evaluation
When you believe your extension is ready:

Submit a claim linking to your public GitHub repository.
The maintainers will independently build your extension from source.
The extension will be evaluated using a clean Chrome installation and a fresh browser profile.
Internet access will be disabled after the initial model download and native localhost APIs will be blocked.
The extension will be tested against the maintainer benchmark using the required 65% confidence threshold.
You will receive the evaluation results.
If your submission reaches the qualifying score and complies with every rule below, you win the bounty.

rules
Your submission may not:

Use cloud inference.
Send image data to external services.
Depend on local Python, Node.js, Flask, or similar backend processes.
Download additional models after initial setup.
Hardcode benchmark image hashes or lookup tables.
Circumvent the evaluation process.
The maintainers reserve the right to disqualify any submission that technically satisfies the letter of these rules while clearly violating their spirit.

The objective of this bounty is to create a practical, easy-to-use, privacy-preserving AI image detector that performs real inference entirely inside a Chrome extension.

winner-take-all
The first submission that legitimately achieves 75.0% balanced accuracy wins the entire prize pool.

A qualifying submission must:

Meet or exceed the required score.
Build successfully from source.
Be fully reproducible.
Comply with every rule above.
Be licensed under the MIT License.
If multiple qualifying submissions are received before verification is complete, priority will be given to the earliest valid submission.

Special Note

If, after significant community participation, it becomes clear that the qualifying threshold is unrealistically high, the bounty creator and contributors may collectively amend the required score before any winner is declared. In this case, we would likely choose to reward the best performing extension that meets the spirit of the bounty.

However, we also reserve the right to cancel this bounty (and refund all contributors) should the benchmark not be met and it is clear that there will not be an extension that comes close to meeting the rest of the bounty requirements.

open source license
All submissions must be released under the MIT License.

The purpose of this bounty is not simply to reward one implementation—it is to create open infrastructure that anyone can study, improve, and build upon.

why this matters
Most "AI-powered" browser tools aren't actually running AI in your browser—they upload your data to a remote server for inference. That means every image you check is shared with a third party.

This bounty asks a harder question: Can we build a detector that's genuinely useful while keeping every image entirely on the user's device?

The challenge extends beyond AI image detection. Modern browser technologies like WebGPU and WebAssembly have only recently made local inference practical. Solving this problem could establish techniques for a new generation of fast, private, browser-native AI applications.

If successful, the result will be an open-source foundation for privacy-preserving AI image detection that anyone can use and improve.
