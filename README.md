# UGFYP_Zero-bit-Watermarking-for-Printed-Image-Authentication
Code is operated in MATLAB.
This project is divided into two parts: watermark generation and ID authentication.
For the first part, users can upload their fingerprint and images to generate an encrypted watermark and store it in the database of MATLAB, and then users will get nums which they can extract the original image and watermark from the database when authentication needed.
For the second part, if users want to authenticate their ownership, only need to enter their nums. The system simulateS the attack combination to evaluate the robutsness to all specifed attacks.

ABSTRACT:
Zero-bit watermarking protects the copyright of the author while ensuring image quality. This paper proposes a DWT-SVD(Discrete Wavelet Transform-Singular Value Decomposition) based scheme that completely avoids any form of distortion of the original image. Designing the system to remain robust to attacks that printed images may suffer and complete copyright identity authentication is one of the most challenging problems owing to the diversity and uncertainty of attacks. The proposed SIFT(Scale Invariant Feature Transform) based algorithm is used to correct the attacked images from geometric attacks(rotation, scaling). Furthermore, the proposed USM(UnSharp Masking) based algorithm is used to correct images from non-geometric attacks(JPEG compressions, tone mapping, illumination). The test results combined with the matching rate of NCC(Normalized Cross-Correlation) show that this system maintains good robustness in most cases. Compared with traditional visible watermarking and other baseline methods, this non-blind watermarking scheme achieves higher security and concealment and also has a better robustness in usage scenarios. Therefore, this is of great significance for users to still be able to complete identity authentication after the image has been printed and recaptured by the camera.
