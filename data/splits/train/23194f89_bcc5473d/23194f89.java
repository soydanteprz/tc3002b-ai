import java.io.*;
import java.util.*;
public class new1{
	
    
    

	
	public static void main(String[] args)  throws IOException{

		
		BufferedWriter output = new BufferedWriter(new OutputStreamWriter(System.out));
		FastReader s = new FastReader();
		int t = s.nextInt();
		for(int z = 0; z < t; z++) {
			int n = s.nextInt();
			int[] ti = new int[n];
			for(int i = 0; i < n; i++) ti[i] = s.nextInt();
			int[] he = new int[n];
			for(int i = 0; i < n; i++) he[i] = s.nextInt();
			long ans = 0;
			int st = ti[n - 1] - he[n - 1] + 1;
			int en = ti[n - 1];
			//System.out.println(st + " " + en);
			for(int i = n - 2; i >= 0; i--) {
				int st1 = ti[i] - he[i] + 1;
				int en1 = ti[i];
				if(en1 >= st) {
					st = Math.min(st, st1);
				}
				else {
					long a1 = ((en - st + 1L) *  (en - st + 2L)) / 2;
					ans = ans + a1;
					st = st1; en = en1;
					//System.out.println(a1);
				}

			}
			ans = ans + ((en - st + 1L) *  (en - st + 2L)) / 2;
			System.out.println(ans);
			
			
			
			
			
			
		}
	   //output.flush();
	  }
}


 
	
