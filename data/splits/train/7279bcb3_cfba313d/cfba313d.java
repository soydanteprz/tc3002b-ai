import java.io.*;
import java.util.*;


public class C {//Any Class Name
	
	static class Code {
		
		private void solve(InputReader in, OutputWriter out) throws IOException {
			
			ArrayOpn o= new ArrayOpn();
			int t = in.readInt();
			while(t -- > 0) {
			
				int n=in.readInt();
				
				
				int[][] freq= new int[5][n];
				
				for(int i=0; i<n; i++) {
					char s[]= in.readString().toCharArray();
					
					
					for(char e: s) {
						if(e=='a') {
							freq[0][i]++;
						}
						else {
							freq[0][i]--;
						}
						if(e=='b') {
							freq[1][i]++;
						}
						else {
							freq[1][i]--;
						}
						
						if(e=='c') {
							freq[2][i]++;
						}
						else {
							freq[2][i]--;
						}
						if(e=='d') {
							freq[3][i]++;
						}
						else {
							freq[3][i]--;
						}
						if(e=='e') {
							freq[4][i]++;
						}
						else {
							freq[4][i]--;
						}
							
					}
				}
				
				
				for(int i=0; i<5; i++) {
					Arrays.sort(freq[i]);
				}
				
				int ans=0;
				for(int i=0; i<5; i++) {
					int temp=0, count=0;
					for(int j=n-1; j>=0; j--) {
						temp+=freq[i][j];
						if(temp<=0) {
							break;
						}
						count++;
					}
					ans=Math.max(ans, count);
				}
				
				out.printLine(ans);
				
			}
			out.close();
		}
	}
	
	static class ArrayOpn{
		
		
		static final Random random = new Random();
		
		private void sort(int A[]) {
			int n = A.length;
			for(int i=0; i<n; i++)
			{
				int j = random.nextInt(n),temp = A[j];
				A[j] = A[i];
				A[i] = temp;
			}
			Arrays.sort(A);
//            return A;
		}
		
		
		
		
		
		
		
		
		
		
		
		
		
	}
	
	
	
	// Comparator for using in Sorting On integers
	
	
	
	
	public static void main(String[] args) throws IOException {
		//initialize
		InputStream inputStream = System.in;
		OutputStream outputStream = System.out;
		InputReader in = new InputReader(inputStream);
		OutputWriter out = new OutputWriter(outputStream);
		Code solver = new Code();
		solver.solve(in, out);
		out.flush();
		out.close();
	}
	
	
	
	
	
	
	
	
	
	
	
}
