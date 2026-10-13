# AI Usage Declaration

This assignment was completed by **Group S** with the support of AI tools.
In accordance with [DSV's AI policy](https://www.su.se/english/divisions/department-of-computer-and-systems-sciences/education/during-your-studies/dsvs-ai-policy),
we declare this use transparently and describe below how the tools were
employed and how their output was integrated.

**AI tool used:** Vibe (Mistral AI), an agentic AI assistant with a sandboxed
execution environment.

## How the AI tool was employed
   Area | What the AI did | How the group integrated and verified it |
 | --- | --- | --- |
 | Planning phase | Discussed the assignment sheet, explained the service/Docker/CI pipeline context and how the components relate | Discussed and confirmed the understanding before starting |
 | Coding | Helped with drafting, syntax, and clearer comments in `train.py`, `app.py`, tests, `Dockerfile`, GitHub Actions workflows, `README.md`, `CHANGELOG.md` | The group reviewed, adapted, committed, and pushed all files; the group owns every design decision in the final code |
 | Model selection (v0.2) | Helped with the experiment scripts (repeated cross-validation comparing Ridge, Lasso, RandomForest, GradientBoosting, polynomial features, PLS) | The results were reproduced independently: the same training code runs in CI and inside the Docker build with pinned dependencies, producing the documented metrics |
 | Debugging | Analyzed CI failure screenshots and proposed fixes | The group applied the fixes locally, pushed, and confirmed the workflows turned green |
 | Verification | Computed expected endpoint values (e.g. the sample-payload predictions 235.95 / 226.913) from the same training code | The group verified the live behavior with `curl` against the running Docker containers |
 | Documentation | Drafted readable comments and corrected spelling | The group reviewed and corrected the content; the group is responsible for its accuracy |

## Work performed by the group itself

- Repository setup, all commits, pull requests, merges, and tags (via GitHub Desktop)
- Local setup and execution of the full pipeline on the group's machines
- Docker acceptance testing (`docker pull`/`docker run`, `curl` against
  `/health`, `/predict`, and the 422 invalid-input case)
- Review, correction, and approval of every AI-supported draft before it was committed

## Examples of the questions asked

- "Before we start, can we talk about the assignment to make sure we understand what we should do?" (understanding the assignment)
- "Can you explain exactly how those POST/GET methods are related to Docker and GitHub Actions?" (architecture)
- "Look at this, do you have an explanation for what went wrong?" + screenshot (iterative debugging of the CI pipeline)
- "We are happy with these results, any last tips on things we might have missed?" (verification of release and acceptance results)

## Responsibility

Per DSV's AI policy, the group bears complete responsibility for the
originality and content of this submission. All AI-generated material was
reviewed, adapted where needed, and verified by the group members before
submission. The final submission is the group's own work product under the
course's examination rules.