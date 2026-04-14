# Makefile for zipping directories and debugging

# Output zip file name
ZIP_NAME := project_archive.zip

# Debug/unzip directory
DEBUG_DIR := debug_unzip

# List of directories to zip
DIRS := src pipelines config plots

# Exclusions for each directory
# Format: EXCLUDE_dir = pattern1 pattern2 ...
EXCLUDE_dir1 := __pycache__ *.log
EXCLUDE_dir2 :=
EXCLUDE_dir3 :=
EXCLUDE_dir4 :=
EXCLUDE_dir5 :=

# Helper function to generate --exclude arguments
define gen_exclude
	$(foreach pattern,$(1),--exclude=$(pattern))
endef

.PHONY: all zip unzip clean_debug clean

all: zip

# Zip directories with exclusions
zip:
	@echo "Creating zip: $(ZIP_NAME)"
	@rm -f $(ZIP_NAME)
	@$(foreach dir,$(DIRS), \
		echo "Adding directory: $(dir)"; \
		zip -r $(ZIP_NAME) $(dir) $(call gen_exclude,$(EXCLUDE_$(dir))) > /dev/null; \
	)
	@echo "Zip creation complete: $(ZIP_NAME)"

# Unzip into debug directory
unzip:
	@echo "Unzipping $(ZIP_NAME) into $(DEBUG_DIR)"
	@mkdir -p $(DEBUG_DIR)
	@unzip -q $(ZIP_NAME) -d $(DEBUG_DIR)
	@echo "Contents available in $(DEBUG_DIR)"

# Remove the zip and debug directory
clean_debug: clean
	@echo "Removing debug directory: $(DEBUG_DIR)"
	@rm -rf $(DEBUG_DIR)

# Remove only the zip file
clean:
	@echo "Removing zip file: $(ZIP_NAME)"
	@rm -f $(ZIP_NAME)


# Directory containing all skills
SKILLS_DIR := .gemini/skills

# Enable a single skill and disable the rest
# Usage: make enable_skill SKILL=python-pro
.PHONY: enable_skill
enable_skill:
ifndef SKILL
	$(error SKILL variable not set. Usage: make enable_skill SKILL=<skill_name>)
endif
	@python3 enable_skill.py $(SKILL)



	