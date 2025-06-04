import java.io.*;
import java.util.*;
import static java.lang.Math.*;

public class E{
	public static void main(String[] args) throws IOException {
		// br = new BufferedReader(new FileReader(".in"));
		// out = new PrintWriter(new FileWriter(".out"));
		//new Thread(null, new (), "peepee", 1<<28).start();
		
		int q =readInt();
		while(q-->0) {
			br.readLine();
			int n =readInt();
			int k =readInt();
			int[] ans = new int[n];
			Arrays.fill(ans, (int)2e9+1);
			int[] a = new int[k];
			int[] t = new int[k];
			int mini = 0, maxi = 0;
			for (int i = 0 ;i  <k; i+=1) {
				a[i]=readInt()-1;
				if (a[i] < a[mini]) mini = i;
				if (a[i] > a[maxi]) maxi = i;
			}
			for (int j = 0; j  <k; j++) t[j]=readInt();
			for (int i = 0; i < k; i++) ans[a[i]]=t[i];
			int[] l = new int[n];
			int[] r = new int[n];
			Arrays.fill(l, (int)2e9);
			Arrays.fill(r, (int)2e9);
			int temp = t[mini];
			for (int i = a[mini]; i < n; i++) {
				l[i] = temp = min(temp+1, ans[i]);
			}
			temp = t[maxi];
			for (int i = a[maxi]; i >= 0; i--) {
				r[i] = temp = min(temp+1,ans[i]);
			}
			
			for (int i = 0; i < n; i++) out.print(min(l[i],r[i]) + " ");
			out.println();
		}
		
		out.close();
	}
	
	static BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
	static PrintWriter out = new PrintWriter(new BufferedOutputStream(System.out));
	static StringTokenizer st = new StringTokenizer("");
	static String read() throws IOException{return st.hasMoreTokens() ? st.nextToken():(st = new StringTokenizer(br.readLine())).nextToken();}
	static int readInt() throws IOException{return Integer.parseInt(read());}
	
	
	
}
