from datasets import load_dataset
import google.generativeai as genai
import time
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image
import io
import re

class GemmaMultimodalBenchmark:

    # Initialize with API key and model configurations
    def __init__(self, api_key=None):
        self.models = {
            "gemma-3-4b-it": "models/gemma-3-4b-it",
            "gemma-3-12b-it": "models/gemma-3-12b-it", 
            "gemma-3-27b-it": "models/gemma-3-27b-it"
        }
        self.results = []
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key is None:
                raise ValueError("Set 'GOOGLE_API_KEY' environment variable or provide it directly.")
        genai.configure(api_key=api_key)
        self.clients = {name: genai.GenerativeModel(model_id) for name, model_id in self.models.items()}
    

    # Resize image for Gemini API
    def prepare_image_for_gemini(self, pil_image):
        if pil_image.mode in ('RGBA', 'P'):
            pil_image = pil_image.convert('RGB')
        max_size = (1024, 1024)
        if pil_image.size[0] > max_size[0] or pil_image.size[1] > max_size[1]:
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return pil_image
    

    # Create prompts based on EMMA dataset strategies provided in the research paper
    def create_prompts(self, question, options, prompt_type, strategy):
        if prompt_type == "multiple_choice":
            if strategy == "CoT":
                return f"{question} {options}\nAnswer with the option's letter from the given choices and put the letter in one \"\\boxed{{}}\". Please solve the problem step by step."
            elif strategy == "Direct":
                return f"{question} {options}\nAnswer with the option's letter from the given choices and put the letter in one \"\\boxed{{}}\". Please ensure that your output only contains the final answer without any additional content (such as intermediate reasoning steps)."
        elif prompt_type == "open_ended":
            if strategy == "CoT":
                return f"{question}\nAnswer the question using a single word or phrase and put the answer in one \"\\boxed{{}}\". Please solve the problem step by step."
            elif strategy == "Direct":
                return f"{question}\nAnswer the question using a single word or phrase and put the answer in one \"\\boxed{{}}\". Please ensure that your output only contains the final answer without any additional content (such as intermediate reasoning steps)."
        return None
    

    # Extract answer from model response
    def extract_answer_from_response(self, response):
        if not response:
            return None
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        boxed_match = re.search(boxed_pattern, response)
        if boxed_match:
            answer = boxed_match.group(1).strip()
            print(f"    Found in \\boxed{{}}: '{answer}'")
            return answer
        letter_patterns = [
            r'[\(\[]\s*([ABCDEF])\s*[\)\]]',
            r'\b([ABCDEF])\b(?!\w)',      
            r'option\s+([ABCDEF])\b',   
            r'choice\s+([ABCDEF])\b',  
            r'answer\s+is\s+([ABCDEF])\b',  
        ]
        response_upper = response.upper()
        for pattern in letter_patterns:
            match = re.search(pattern, response_upper)
            if match:
                answer = match.group(1)
                print(f"    Found with pattern '{pattern}': '{answer}'")
                return answer
        letter_match = re.search(r'\b([ABCDEF])\b', response_upper)
        if letter_match:
            answer = letter_match.group(1)
            print(f"    Found letter in text: '{answer}'")
            return answer
        print(f"    No answer pattern found in response")
        return None
    

    # Check if predicted answer matches actual answer
    def is_answer_correct(self, predicted_answer, ground_truth):
        if not predicted_answer or not ground_truth:
            return False
        pred_norm = str(predicted_answer).strip().upper()
        truth_norm = str(ground_truth).strip().upper()
        print(f"    Comparing: '{pred_norm}' vs '{truth_norm}'")
        if pred_norm == truth_norm:
            return True
        if truth_norm in pred_norm:
            return True
        return False


    # Query Gemma model with images and text
    def query_gemma_multimodal(self, model_name, images, prompt, max_retries=3):
        prepared_images = [self.prepare_image_for_gemini(img) for img in images if img is not None]
        if not prepared_images:
            return {
                "response": "Error: No valid images provided",
                "processing_time": 0,
                "success": False
            }
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.clients[model_name].generate_content([prompt] + prepared_images)
                end_time = time.time()
                if response and response.text:
                    return {
                        "response": response.text.strip(),
                        "processing_time": end_time - start_time,
                        "success": True
                    }
                else:
                    return {
                        "response": "Error: Empty response from model",
                        "processing_time": end_time - start_time,
                        "success": False
                    }
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed for {model_name}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue
        return {
            "response": f"Error: All attempts failed",
            "processing_time": 0,
            "success": False
        }
    

    # Run EMMA benchmark across multiple prompt strategies and models
    def run_benchmark(self, num_samples=50):
        print("Loading EMMA-mini Coding dataset...")
        ds = load_dataset("luckychao/EMMA-mini", "Coding")
        split_data = ds['test']
        num_samples = min(num_samples, len(split_data))
        print(f"Testing Gemma multimodal models on {num_samples} samples...")
        print("Available models:", list(self.models.keys()))
        prompt_strategies = [
            ("multiple_choice", "CoT"),
            ("multiple_choice", "Direct"),
            ("open_ended", "CoT"), 
            ("open_ended", "Direct")
        ]
        for sample_idx in tqdm(range(num_samples)):
            sample = split_data[sample_idx]
            images = []
            for i in range(1, 6):
                image_key = f'image_{i}'
                if image_key in sample and sample[image_key] is not None:
                    images.append(sample[image_key])
            if not images:
                print(f"Skipping sample {sample_idx}: No images found")
                continue
            question = sample.get('question', '')
            answer = sample.get('answer', '')
            options = sample.get('options', '')
            category = sample.get('category', 'Unknown')
            task = sample.get('task', 'Unknown')
            print(f"\nSample {sample_idx}: {len(images)} images | Category: {category} | Task: {task}")
            print(f"Question: {question[:100]}...")
            print(f"Correct answer: {answer}")
            print(f"Options: {options}")
            
            for prompt_type, strategy in prompt_strategies:
                prompt = self.create_prompts(question, options, prompt_type, strategy)
                if not prompt:
                    continue
                print(f"\n  Strategy: {prompt_type} - {strategy}")
                print(f"  Prompt: {prompt[:100]}...")
                for model_name in self.models.keys():
                    print(f"    Testing {model_name}...")
                    result = self.query_gemma_multimodal(model_name, images, prompt)
                    predicted_answer = None
                    is_correct = False
                    if result['success']:
                        predicted_answer = self.extract_answer_from_response(result['response'])
                        is_correct = self.is_answer_correct(predicted_answer, answer)
                    self.results.append({
                        'sample_id': sample_idx,
                        'model': model_name,
                        'prompt_type': prompt_type,
                        'strategy': strategy,
                        'category': category,
                        'task': task,
                        'images_count': len(images),
                        'processing_time': result['processing_time'],
                        'success': result['success'],
                        'predicted_answer': predicted_answer,
                        'ground_truth_answer': answer,
                        'is_correct': is_correct,
                        'predicted_response': result['response'],
                        'dataset_question': question,
                        'dataset_options': options
                    })
                    if result['success']:
                        status = "CORRECT" if is_correct else "WRONG"
                        print(f"      {status} | Predicted: {predicted_answer} | Ground truth: {answer}")
                        print(f"        Time: {result['processing_time']:.2f}s")
                    else:
                        print(f" Failed")
                    time.sleep(1)
        return self.results
    

    # Save results to CSV
    def save_results(self, filename="gemma_prompt_strategies_benchmark.csv"):
        if not self.results:
            print("No results to save.")
            return None
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        return df
    

    # Generate report
    def generate_comprehensive_report(self):
        if not self.results:
            print("No results to analyze.")
            return
        df = pd.DataFrame(self.results)
        successful_df = df[df['success']]
        print("\n" + "="*120)
        print("Comprehensive Report")
        print("="*120)
        print(f"\n Overall Results:")
        strategies = [('multiple_choice', 'CoT'), ('multiple_choice', 'Direct'), 
                     ('open_ended', 'CoT'), ('open_ended', 'Direct')]
        print(f"\n{'MODEL':<20} {'MC-CoT':<12} {'MC-Direct':<12} {'OE-CoT':<12} {'OE-Direct':<12} {'OVERALL':<12}")
        print(f"{'-'*90}")

        for model in self.models.keys():
            model_data = successful_df[successful_df['model'] == model]
            row = f"{model:<20}"
            overall_correct = 0
            overall_total = 0
            for prompt_type, strategy in strategies:
                strategy_data = model_data[
                    (model_data['prompt_type'] == prompt_type) & 
                    (model_data['strategy'] == strategy)
                ]
                if len(strategy_data) > 0:
                    correct = strategy_data['is_correct'].sum()
                    total = len(strategy_data)
                    accuracy = correct / total if total > 0 else 0
                    overall_correct += correct
                    overall_total += total
                    row += f"{accuracy:.1%} ({correct}/{total})"
                else:
                    row += f"{'N/A':<12}"
                row += " "
            overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
            row += f"{overall_accuracy:.1%} ({overall_correct}/{overall_total})"
            print(row)
        print(f"\n\n🔍 Model wise details")
        
        for model in self.models.keys():
            print(f"\n{'-'*60}")
            print(f"MODEL: {model}")
            print(f"{'-'*60}")
            model_data = successful_df[successful_df['model'] == model]
            total_correct = model_data['is_correct'].sum()
            total_questions = len(model_data)
            overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
            print(f"Overall: {overall_accuracy:.1%} ({total_correct}/{total_questions})")
            print(f"\nBy Strategy:")
            for prompt_type, strategy in strategies:
                strategy_data = model_data[
                    (model_data['prompt_type'] == prompt_type) & 
                    (model_data['strategy'] == strategy)
                ]
                if len(strategy_data) > 0:
                    correct = strategy_data['is_correct'].sum()
                    total = len(strategy_data)
                    accuracy = correct / total
                    avg_time = strategy_data['processing_time'].mean()
                    print(f"  {prompt_type}-{strategy}: {accuracy:.1%} ({correct}/{total}) | Avg time: {avg_time:.2f}s")
        print(f"\n\n Chain-of-Thought vs Direct Comparison")
        print(f"{'MODEL':<20} {'CoT Accuracy':<15} {'Direct Accuracy':<15} {'Difference':<12}")
        print(f"{'-'*65}")
        
        for model in self.models.keys():
            model_data = successful_df[successful_df['model'] == model]
            cot_data = model_data[model_data['strategy'] == 'CoT']
            cot_accuracy = cot_data['is_correct'].mean() if len(cot_data) > 0 else 0
            direct_data = model_data[model_data['strategy'] == 'Direct']
            direct_accuracy = direct_data['is_correct'].mean() if len(direct_data) > 0 else 0
            difference = cot_accuracy - direct_accuracy
            difference_str = f"+{difference:.3f}" if difference > 0 else f"{difference:.3f}"
            print(f"{model:<20} {cot_accuracy:.1%}{'':<8} {direct_accuracy:.1%}{'':<8} {difference_str}")
        if 'category' in successful_df.columns and successful_df['category'].notna().any():
            print(f"\n\n Category-wise Performance of Top 5")
            categories = successful_df['category'].value_counts().head(5).index
            for category in categories:
                print(f"\nCategory: {category}")
                category_data = successful_df[successful_df['category'] == category]          
                for model in self.models.keys():
                    model_cat_data = category_data[category_data['model'] == model]
                    if len(model_cat_data) > 0:
                        accuracy = model_cat_data['is_correct'].mean()
                        count = len(model_cat_data)
                        print(f"  {model}: {accuracy:.1%} ({count} samples)")


# Set your API key and run the benchmark
def main():
    API_KEY = ""
    print("=== GEMMA MULTIMODAL MODELS - PROMPT STRATEGY BENCHMARK ===")
    print("Testing: gemma-3-4b-it, gemma-3-12b-it, gemma-3-27b-it")
    print("Prompt Strategies: Multiple-Choice (CoT/Direct), Open-Ended (CoT/Direct)")
    print("Dataset: 100 Coding Questions from EMMA-mini")
    benchmark = GemmaMultimodalBenchmark(api_key=API_KEY)
    print("Starting benchmark with 100 samples...")
    results = benchmark.run_benchmark(num_samples=100)
    benchmark.save_results("gemma_prompt_strategies_benchmark.csv")
    benchmark.generate_comprehensive_report()
if __name__ == "__main__":
    main()
