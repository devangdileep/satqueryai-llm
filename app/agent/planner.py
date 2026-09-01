from app.schemas.analysis import QueryAnalysis
from app.schemas.images import ImageInputConfig
from app.schemas.models import ModelSelection
from app.schemas.jobs import WorkflowPlan, WorkflowStep


class WorkflowPlanner:
    """Generates an explicit, observable execution plan for the SatQuery agent."""

    def plan_workflow(
        self,
        analysis: QueryAnalysis,
        input_config: ImageInputConfig,
        model_selection: ModelSelection
    ) -> WorkflowPlan:
        steps: list[WorkflowStep] = []
        step_num = 1

        # Step 1: Input Validation
        steps.append(
            WorkflowStep(
                step_number=step_num,
                tool="validate_image",
                description="Validate image format, dimensions, band structure, CRS, and geotransform.",
                inputs={"image_paths": [img.filename for img in input_config.images]}
            )
        )
        step_num += 1

        # Step 2: Metadata & Modality Classification
        steps.append(
            WorkflowStep(
                step_number=step_num,
                tool="classify_image_modality",
                description="Determine satellite image modality (optical, multispectral, SAR) and sensor characteristics.",
                inputs={"images": [img.filename for img in input_config.images]}
            )
        )
        step_num += 1

        # Step 3: Specialist Model Inference
        selected_model = model_selection.selected_model.lower()
        steps.append(
            WorkflowStep(
                step_number=step_num,
                tool=selected_model,
                description=f"Execute specialist remote sensing model: {model_selection.selected_model} for task {analysis.task}.",
                inputs={
                    "model": model_selection.selected_model,
                    "task": analysis.task,
                    "query": analysis.raw_query
                }
            )
        )
        step_num += 1

        # Step 4: Region Grounding if requested
        if analysis.requires_spatial_evidence or "grounding" in analysis.intent:
            steps.append(
                WorkflowStep(
                    step_number=step_num,
                    tool="region_grounding",
                    description="Ground detected targets and render bounding box overlays on imagery.",
                    inputs={"target_entities": analysis.target_entities}
                )
            )
            step_num += 1

        # Step 5: Visual Evidence Generation
        steps.append(
            WorkflowStep(
                step_number=step_num,
                tool="generate_visual_evidence",
                description="Package visual evidence, overlays, and change maps.",
                inputs={"task": analysis.task}
            )
        )
        step_num += 1

        # Step 6: Answer Composition
        steps.append(
            WorkflowStep(
                step_number=step_num,
                tool="compose_answer",
                description="Synthesize grounded observations, evidence, and confidence into final answer response.",
                inputs={"query": analysis.raw_query}
            )
        )

        return WorkflowPlan(
            task=analysis.task,
            steps=steps,
            estimated_duration_ms=1200.0
        )


workflow_planner = WorkflowPlanner()
