from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
from pathlib import Path
import litellm

load_dotenv()
litellm.set_verbose = True

pdf_path = str(Path(__file__).parent.parent.parent) + '/knowledge/rfc2326.pdf'
rag_tool = PDFSearchTool(pdf=pdf_path)

@CrewBase
class RagEvaluation():
	"""RagEvaluation crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def state_machine_specialist(self) -> Agent:
		return Agent(
			config=self.agents_config['state_machine_specialist'],
   			tools=[rag_tool],
			verbose=True,
			max_retry_limit=3,
		)
  
	@task
	def state_machine_task(self) -> Task:
		return Task(
			config=self.tasks_config['state_machine_task'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the RagEvaluation crew"""
	
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
