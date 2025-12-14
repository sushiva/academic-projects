import re

# Read the current presentation
with open('presentation_content.md', 'r') as f:
    content = f.read()

# Add images to Slide 6 (EDA)
content = re.sub(
    r'(\*\*Include visualization:\*\* Grid showing sample images from both classes)',
    r'![Class Distribution](../outputs/class_distribution.png)\n\n![Samples With Helmet](../outputs/samples_with_helmet.png)\n\n![Samples Without Helmet](../outputs/samples_without_helmet.png)',
    content
)

# Add image to Slide 8 (Model 1)
content = re.sub(
    r'(### Key Observations\n- \*\*Perfect baseline performance)',
    r'![Model 1 Training History](../outputs/training/plots/model1_simple_cnn/training_history.png)\n\n\1',
    content
)

# Add image to Slide 9 (Model 2)
content = re.sub(
    r'(\*\*Critical Insight:\*\* Transfer learning is incredibly powerful for image classification)',
    r'![Model 2 Training History](../outputs/training/plots/model2_resnet_base/training_history.png)\n\n\1',
    content
)

# Add image to Slide 10 (Model 3)
content = re.sub(
    r'(### Key Observations\n- \*\*417x more trainable parameters)',
    r'![Model 3 Training History](../outputs/training/plots/model3_resnet_deep/training_history.png)\n\n\1',
    content
)

# Add image to Slide 11 (Model 4)
content = re.sub(
    r'(### Key Observations\n- \*\*11,388x more trainable parameters)',
    r'![Model 4 Training History](../outputs/training/plots/model4_resnet_augmented/training_history.png)\n\n\1',
    content
)

# Add images to Slide 12 (Comparison)
content = re.sub(
    r'(\*\*Include visualization:\*\*\n- Bar chart comparing trainable parameters\n- Line plots showing learning curves for all 4 models\n- Confusion matrices \(all should be perfect diagonals\))',
    r'![Parameter Comparison](../outputs/comparison/model_comparison_bars.png)\n\n![Progressive Improvement](../outputs/comparison/progressive_improvement.png)\n\n![Confusion Matrices Grid](../outputs/comparison/confusion_matrices_grid.png)',
    content
)

# Write updated content
with open('presentation_content_with_images.md', 'w') as f:
    f.write(content)

print("Helmet presentation updated with images!")
