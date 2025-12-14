import re

# Read the current presentation
with open('presentation_content.md', 'r') as f:
    content = f.read()

# Add image to Slide 8 (Model 1)
content = re.sub(
    r'(### Key Observations\n- \*\*Best performing model)',
    r'![Model 1 Training History](../outputs/training/plots/model1_ann_rgb_history.png)\n\n![Model 1 Confusion Matrix](../outputs/evaluation/model1_ann_rgb_confusion_matrix.png)\n\n\1',
    content
)

# Add image to Slide 9 (Model 2)
content = re.sub(
    r'(### Key Observations\n- \*\*Unexpected underperformance)',
    r'![Model 2 Training History](../outputs/training/plots/model2_ann_grayscale_history.png)\n\n![Model 2 Confusion Matrix](../outputs/evaluation/model2_ann_grayscale_confusion_matrix.png)\n\n\1',
    content
)

# Add image to Slide 10 (Model 3)
content = re.sub(
    r'(### Key Observations\n- \*\*Severe performance degradation)',
    r'![Model 3 Training History](../outputs/training/plots/model3_ann_blur_history.png)\n\n![Model 3 Confusion Matrix](../outputs/evaluation/model3_ann_blur_confusion_matrix.png)\n\n\1',
    content
)

# Add image to Slide 11 (Model 4)
content = re.sub(
    r'(### Key Observations\n- \*\*Extreme overfitting)',
    r'![Model 4 Training History](../outputs/training/plots/model4_ann_laplacian_history.png)\n\n![Model 4 Confusion Matrix](../outputs/evaluation/model4_ann_laplacian_confusion_matrix.png)\n\n\1',
    content
)

# Add image to Slide 12 (Comparison)
content = re.sub(
    r'(\*\*Include visualization:\*\* Bar chart comparing test accuracies of all 4 models)',
    r'![Model Comparison](../outputs/comparison/model_comparison.png)',
    content
)

# Write updated content
with open('presentation_content_with_images.md', 'w') as f:
    f.write(content)

print("COVID presentation updated with images!")
