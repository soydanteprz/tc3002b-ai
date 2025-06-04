import java.io.*;
import java.util.*;

public class B {
	
	public static void main(String[] args)throws IOException {
		
		FastScanner scan = new FastScanner();
		PrintWriter output = new PrintWriter(System.out);
		int t = scan.nextInt();
		for(int tt = 0;tt<t;tt++) {
			int n = scan.nextInt();
			char initial[] = scan.next().toCharArray();
			char desired[] = scan.next().toCharArray();
			int lit1 = 0, lit2 = 0;
			int ans = Integer.MAX_VALUE;
			for(int i = 0;i<n;i++) {
				if(initial[i]=='1') lit1++;
				if(desired[i]=='1') lit2++;
			}
			if(lit1==lit2) {
				int count = 0;
				for(int i = 0;i<n;i++) if(initial[i]!=desired[i]) count++;
				ans = Math.min(ans, count);
			}
			if(lit2==(n-lit1+1)) {
				int count = 0;
				for(int i = 0;i<n;i++) if(initial[i]==desired[i]) count++;
				ans = Math.min(ans, count);
			}
			if(ans == Integer.MAX_VALUE) ans = -1;
			output.println(ans);	
		}
		output.flush();
	}
	
	
	
	
	
	
	
	

}
