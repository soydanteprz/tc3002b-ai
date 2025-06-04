import java.io.*;
import java.lang.*;
import java.util.*;
public class MinGridPath {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner s=new Scanner(System.in);
		int t=s.nextInt();
		while(t-->0){
			
			int n = s.nextInt();
			long[] aa =new long[n];
			for(int i=0;i<n;i++)
				aa[i]=s.nextLong();
 
			long minEven = aa[0];
			long minOdd = aa[1];
			long sum = aa[0]+aa[1];
			long best = n*minEven + n*minOdd;
			int numOdd = 1;
			int numEven = 1;
			for(int i=2; i<n; ++i) {
				if(i%2 == 0) {
					minEven = Math.min(aa[i], minEven);
					numEven++;
				}else {
					minOdd = Math.min(aa[i], minOdd);
					numOdd++;
				}
				sum += aa[i];
				long score = sum;
				score += minEven*(n-numEven);
				score += minOdd*(n-numOdd);				
				best = Math.min(best, score);
			}
			System.out.println(best);
			
		}
	}

}
