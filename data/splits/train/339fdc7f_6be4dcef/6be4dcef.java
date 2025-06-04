import java.util.*;
import java.io.*;
public class ProbC {

	public static void main(String[] args) throws IOException {
		
		BufferedReader bf = new BufferedReader(new InputStreamReader(System.in));
		StringTokenizer st;
		
		StringBuilder sb = new StringBuilder();
		
		int cases = Integer.parseInt(bf.readLine());
		
		for(int caseNum=0; caseNum<cases; caseNum++) {
			
			int n = Integer.parseInt(bf.readLine());
			
			String[] words = new String[n];
			
			for(int i=0; i<n; i++)
				words[i] = bf.readLine();
			
			int[] count = new int[26];
			
			for(int i=0; i<n; i++)
				for(int j=0; j<words[i].length(); j++)
					count[words[i].charAt(j)-97]++;
			
			int maxWords = 0;
			
			for(int x=0; x<26; x++) {
				
				if(count[x] > 0) {
					
					char curChar = (char)(x+97);
			
					int[] diff = new int[n];
					
					for(int i=0; i<n; i++) {
						for(int j=0; j<words[i].length(); j++) {
							if(words[i].charAt(j) == curChar)
								diff[i]++;
							else
								diff[i]--;
						}
					}
					
					Arrays.sort(diff);
					
					int sum = 0;
					
					int i = diff.length-1;
					
					//System.out.println(Arrays.toString(diff)+" "+curChar);
					
					for(; i>=0; i--) {
						sum+=diff[i];
						if(sum <= 0)
							break;
					}
					
					maxWords = Math.max(maxWords, n-(i+1));
				
				}
			
			}
			
			sb.append(maxWords+"\n");
			
		}
		
		System.out.println(sb);
		
		bf.close();

	}

}
