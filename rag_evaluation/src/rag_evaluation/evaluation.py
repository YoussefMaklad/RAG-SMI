import ast
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Headers and attributes for RTSP requests
requestHeaders = [
    "ANNOUNCE",
    "SETUP",
    "DESCRIBE",
    "OPTIONS",
    "PAUSE",
    "PLAY",
    "RECORD",
    "REDIRECT",
    "TEARDOWN",
    "GET_PARAMETER",
    "SET_PARAMETER",
]
requestAttributes = [
    "CSeq",
    "User-Agent",
    "Session",
    "Content-Type",
    "Content-Length",
    "Transport",
    "Accept",
    "Range",
    "Scale",
    "Speed",
    "Location"
]
rtsp_requests = {
    "ANNOUNCE": ["CSeq", "User-Agent", "Session", "Content-Type", "Content-Length"],
    "SETUP": ["CSeq", "User-Agent", "Transport"],
    "DESCRIBE": ["CSeq", "User-Agent", "Accept"],
    "OPTIONS": ["CSeq", "User-Agent"],
    "PAUSE": ["CSeq", "User-Agent", "Session"],
    "PLAY": ["CSeq", "User-Agent", "Session", "Range", "Scale", "Speed"],
    "RECORD": ["CSeq", "User-Agent", "Session", "Range"],
    "REDIRECT": ["CSeq", "User-Agent", "Session", "Location"],
    "TEARDOWN": ["CSeq", "User-Agent", "Session"],
    "GET_PARAMETER": ["CSeq", "User-Agent", "Session", "Content-Type", "Content-Length"],
    "SET_PARAMETER": ["CSeq", "User-Agent", "Session", "Content-Type", "Content-Length"],
}
ground_truth ={
  "DESCRIBE": "DESCRIBE rtsp://127.0.0.1:8554/ RTSP/1.0\r\nCSeq: 2\r\nUser-Agent: ./testRTSPClient (LIVE555 Streaming Media v2018.08.28)\r\nAccept: application/sdp\r\n",
  "SETUP": "SETUP rtsp://127.0.0.1:8554//track1 RTSP/1.0\r\nCSeq: 3\r\nUser-Agent: ./testRTSPClient (LIVE555 Streaming Media v2018.08.28)\r\nTransport: RTP/AVP;unicast;client_port=1234-1235\r\n",
  "PLAY": "PLAY rtsp://127.0.0.1:8554/ RTSP/1.0\r\nCSeq: 4\r\nUser-Agent: ./testRTSPClient (LIVE555 Streaming Media v2018.08.28)\r\nSession: 12345678\r\n",
  "PAUSE": "PAUSE rtsp://127.0.0.1:8554/ RTSP/1.0\r\nCSeq: 5\r\nUser-Agent: ./testRTSPClient (LIVE555 Streaming Media v2018.08.28)\r\nSession: 12345678\r\n",
  "TEARDOWN": "TEARDOWN rtsp://127.0.0.1:8554/ RTSP/1.0\r\nCSeq: 6\r\nUser-Agent: ./testRTSPClient (LIVE555 Streaming Media v2018.08.28)\r\nSession: 12345678\r\n",
  "GET_PARAMETER": "GET_PARAMETER rtsp://127.0.0.1:8554/ RTSP/1.0\r\nCSeq: 7\r\nUser-Agent: ./testRTSPClient (LIVE555 Streaming Media v2018.08.28)\r\nSession: 12345678\r\nContent-Type: text/parameters\r\nContent-Length: 14\r\n\r\nparameter-name\r\n",
  "SET_PARAMETER": "SET_PARAMETER rtsp://127.0.0.1:8554/ RTSP/1.0\r\nCSeq: 8\r\nUser-Agent: ./testRTSPClient (LIVE555 Streaming Media v2018.08.28)\r\nSession: 12345678\r\nContent-Type: text/parameters\r\nContent-Length: 21\r\n\r\nparameter-name=value\r\n",
  "ANNOUNCE": "ANNOUNCE rtsp://127.0.0.1:8554/ RTSP/1.0\r\nCSeq: 9\r\nUser-Agent: ./testRTSPClient (LIVE555 Streaming Media v2018.08.28)\r\nContent-Type: application/sdp\r\nContent-Length: 123\r\n\r\nv=0\no=- 0 0 IN IP4 127.0.0.1\ns=Session streamed by 'LIVE555 Media Server'\nt=0 0\na=tool:LIVE555 Streaming Media v2018.08.28\r\n",
  "RECORD": "RECORD rtsp://127.0.0.1:8554/ RTSP/1.0\r\nCSeq: 10\r\nUser-Agent: ./testRTSPClient (LIVE555 Streaming Media v2018.08.28)\r\nSession: 12345678\r\nRange: npt=0-\r\n",
  "REDIRECT": "REDIRECT rtsp://127.0.0.1:8554/ RTSP/1.0\r\nCSeq: 11\r\nUser-Agent: ./testRTSPClient (LIVE555 Streaming Media v2018.08.28)\r\nLocation: rtsp://127.0.0.2:8554/\r\n"
}


def CheckRequestStructure(Request, score):
    # Validate header
    valid = False
    for requestHeader in requestHeaders:
        requestName = re.search(requestHeader, Request[0])
        if requestName:
            valid = True
            requestName = requestName.group()
            break

    if not valid:
        score["incorrect Header"] += 1
        return

    # Check attributes
    Searched = []
    attributes = rtsp_requests[requestName]
    for item in Request[1:]:
        valid = False
        for attribute in attributes:
            if re.search(attribute, item):
                if attribute in Searched:
                    score["repeated Attributes"] += 1
                else:
                    Searched.append(attribute)
                valid = True
                break
        if not valid:
            score["incorrect Attribute"] += 1

    # Check for missing attributes
    missing_attrs = set(attributes) - set(Searched)
    score["missing Attribute"] += len(missing_attrs)

def CheckExtractedGrammer(Requests):
    # Initialize a single score dictionary
    score = {
        "incorrect Header": 0,
        "incorrect Attribute": 0,
        "missing Attribute": 0,
        "repeated Attributes": 0,
    }

    for Request in Requests:
        CheckRequestStructure(Request, score)
    
    return score

def calculate_bleu_scores( candidate_packets):
    """
    Calculate BLEU scores for the given ground truth and candidate packets.
    """
    # Function to tokenize packets (simplified version)
    def tokenize_packet(packet):
        return packet.replace("\r\n", " ").split()

    # Initialize smoothing function for BLEU score
    smoothing_function = SmoothingFunction().method1  # To handle small differences gracefully
    bleu_scores = {}

    for method, ground_truth_packet in ground_truth.items():
        candidate_packet = candidate_packets.get(method)

        if candidate_packet:
            # Tokenize both the reference and candidate packets
            reference_tokens = [tokenize_packet(ground_truth_packet)]
            candidate_tokens = tokenize_packet(candidate_packet)

            # Calculate BLEU score and store it
            score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing_function)
            bleu_scores[method] = score
        else:
            # No candidate packet for the method
            bleu_scores[method] = None

    return bleu_scores

def create_candidate_packets_from_list(raw_requests):
    """
    Create a dictionary of candidate packets from a list of raw request strings.
    """
    candidate_packets = {}

    for request in raw_requests:
        # Extract the method (the first word in the request string)
        method, *rest = request.split()
        cleaned_request = " ".join(rest).replace("\\/", "/").strip()

        # Add cleaned request to candidate packets dictionary
        if method in candidate_packets:
            # If multiple packets for the same method, store them in a list
            if isinstance(candidate_packets[method], list):
                candidate_packets[method].append(cleaned_request)
            else:
                candidate_packets[method] = [candidate_packets[method], cleaned_request]
        else:
            candidate_packets[method] = cleaned_request

    return candidate_packets

def calculate_rouge_scores( candidate_packets):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {}

    for method, ground_truth_packet in ground_truth.items():
        candidate_packet = candidate_packets.get(method)

        if candidate_packet:
            # Calculate ROUGE score for different metrics
            scores = scorer.score(ground_truth_packet, candidate_packet)
            rouge_scores[method] = {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure
            }
        else:
            rouge_scores[method] = None  # No candidate packet provided

    return rouge_scores

def formatGrammerPromptIntoList(text_block):
    """
    Extracts RTSP request blocks from the given text and removes 'RTSP/1.0' if present at the beginning of any request.
    """
    list_of_requests = []
    temp_request = []

    # Iterate over each line
    for line in text_block.splitlines():
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        # Detect new request by matching known headers
        if any(header in line for header in requestHeaders):
            if temp_request:
                list_of_requests.append("\r\n".join(temp_request))
                temp_request = []
            # Remove 'RTSP/1.0 ' from the beginning of the request
            if line.startswith("RTSP/1.0"):
                line = line.replace("RTSP/1.0 ", "", 1)
            temp_request.append(line)
        else:
            temp_request.append(line)

    # Add the last request if present
    if temp_request:
        list_of_requests.append("\r\n".join(temp_request))

    # Post-process each request to remove leftover 'RTSP/1.0' at the start
    list_of_requests = [req.replace("RTSP/1.0 ", "") for req in list_of_requests]

    return list_of_requests



if __name__ == "__main__":
    packet = """DESCRIBE rtsp:\/\/127.0.0.1:8554\/aacAudioTest RTSP\/1.0\r\nCSeq: 2\r\nUser-Agent: .\/testRTSPClient (LIVE555 Streaming Media v2018.08.28)\r\nAccept: application\/sdp\r\n\r\nSETUP rtsp:\/\/127.0.0.1:8554\/aacAudioTest\/track1 RTSP\/1.0\r\nCSeq: 3\r\nUser-Agent: .\/testRTSPClient (LIVE555 Streaming Media v2018.08.28)\r\nTransport: RTP\/AVP;unicast;client_port=38784-38785\r\n\r\nPLAY rtsp:\/\/127.0.0.1:8554\/aacAudioTest\/ RTSP\/1.0\r\nCSeq: 4\r\nUser-Agent: .\/testRTSPClient (LIVE555 Streaming Media v2018.08.28)\r\nSession: 000022B8\r\nRange: npt=0.000-\n
"""
    text1 = """RTSP/1.0 SETUP rtsp://127.0.7.1:8554/aacAudioTest RTSP/1.0
CSeq: 3
User-Agent: ./testRTSPClient
Transport: RTP/AVP;unicast;client_port=9000-9001"""
    
    extracted = formatGrammerPromptIntoList(packet)
    print("----------- Extracted requests: ------------")
    print(extracted)
    print("--------------------------------------------")

    # Create candidate packets
    candidate_packets = create_candidate_packets_from_list(extracted)

    # Calculate BLEU scores
    print("----------- BLEU Score ------------")
    bleu_scores = calculate_bleu_scores( candidate_packets)
    print(bleu_scores)
    print("--------------------------------------------")

    print("----------- ROUGE Score ------------")
    # Call the ROUGE score calculation function
    rouge_scores = calculate_rouge_scores(candidate_packets)
    print(rouge_scores)
    print("--------------------------------------------")

    
    score = CheckExtractedGrammer(extracted)
    print("-------------- Final Validation Score: --------------------")
    print(score)
    print("--------------------------------------------------")
    
    
