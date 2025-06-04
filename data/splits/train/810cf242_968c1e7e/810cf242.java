import java.util.*;
import java.io.*;
public class E1547{
	static int inf = 2*(int)Math.pow(10,9);
	static FastScanner fs = null;
	public static void main(String[] args) {
		fs = new FastScanner();
		int t = fs.nextInt();
		while(t-->0){
			int n = fs.nextInt();
			int k = fs.nextInt();
			int pos[] = new int[k];
			for(int i=0;i<k;i++){
				pos[i] = fs.nextInt();
			}	
			int time[] = new int[n];
			Arrays.fill(time,inf);
			for(int i=0;i<k;i++){
				time[pos[i]-1] = fs.nextInt();
			}
			for(int i=0;i<n;i++){
				if(i==0){
					continue;
				}
				else{
					time[i] = Math.min(time[i],time[i-1]+1);
				}
			}
			for(int i=n-2;i>=0;i--){
				time[i] = Math.min(time[i],1+time[i+1]);
			}
			for(int i=0;i<n;i++){
				System.out.print(time[i]+" ");
			}
			System.out.println();			
		}
	}
	
	
}
