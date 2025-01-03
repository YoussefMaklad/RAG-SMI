import sys
import warnings
import json
import csv
import time
import litellm
from crew import RagEvaluation
from evaluation import calculate_bleu_scores, calculate_rouge_scores, CheckExtractedGrammer, create_candidate_packets_from_list, formatGrammerPromptIntoList

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
litellm.set_verbose=True

log_file = open("evaluation4.log", "a")

def run():
    """
    Run the crew for each entry in the input JSON file.
    Evaluate outputs and save results to a CSV file.
    """
    counter = 0
    
    # Load JSON data
    with open('qa-state-machine.json', 'r') as file:
        data = json.load(file)

    results = []

    for entry in data[111:150]:
        question = entry[0]['value']

        # Process input
        inputs = {
            'protocol': 'RTSP',
            'question': f"""{question}"""
        }

        try:
            # Generate output using crew
            result = RagEvaluation().crew().kickoff(inputs=inputs)
            generated_response = result.raw
            print(f"-------------- Response ----------------: \n\n {generated_response} \n\n ----------------------------------------")

            # Evaluation
            extracted = formatGrammerPromptIntoList(generated_response)
            candidate_packets = create_candidate_packets_from_list(extracted)
            bleu_scores = calculate_bleu_scores(candidate_packets)
            rouge_scores = calculate_rouge_scores(candidate_packets)
            validation_score = CheckExtractedGrammer(extracted)

            log_file.write("-------------------------------------------------------------")
            log_file.write(f"Question: {question} \n\n\n")
            log_file.write(f"Agent Response: {generated_response} \n\n\n")
            log_file.write(f"Extracted Grammer: {extracted} \n\n\n")
            log_file.write(f"BLEU Scores: {bleu_scores} \n\n\n")
            log_file.write(f"ROUGE Scores: {rouge_scores} \n\n\n")
            log_file.write(f"Validation Score: {validation_score} \n\n\n")
            log_file.write("-------------------------------------------------------------")
            
            print("-------------------------------------------------------------")
            print(f"Question: {question} \n")
            print(f"Agent Response: {generated_response} \n")
            print(f"Extracted Grammer: {extracted} \n")
            print(f"BLEU Scores: {bleu_scores} \n")
            print(f"ROUGE Scores: {rouge_scores} \n")
            print(f"Validation Score: {validation_score} \n")
            print("-------------------------------------------------------------")

            # Store results
            results.append({
                'question': question,
                'agent_response': generated_response,
                'bleu_scores': bleu_scores,
                'rouge_scores': rouge_scores,
                'validation_score': validation_score
            })

            counter += 1
            print("----------------- Sleeping for 10 seconds... --------------------")
            log_file.write(f"Processed {counter} questions")
            print(f"Processed {counter} questions")
            time.sleep(10)
            print("----------------- Done Sleeping, Proceeding now... --------------------")
            
        except Exception as e:
            log_file.write(f"Error: {e}")
            log_file.write(f"Error for question: {question}")
            print(f"Error for question: {question}")
            print(f"Error: {e}")
            pass
        except KeyboardInterrupt:
            log_file.write(f"Keyboard Interrupt")
            log_file.write(f"Processed {counter} questions")
            print(f"*"*30)
            print("SAVING RESULTS TO CSV FILE NOW")
            print(f"*"*30)
            break
      

    # Save results to CSV
    with open('evaluation_results4.csv', 'w', newline='') as csvfile:
        fieldnames = ['question', 'agent_response', 'bleu_scores', 'rouge_scores', 'validation_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Evaluation completed and saved to evaluation_results4.csv!")


if __name__ == '__main__':
    run()