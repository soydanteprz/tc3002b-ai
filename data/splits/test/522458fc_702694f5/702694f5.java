import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
//import java.text.DecimalFormat;
import java.util.*;

//import B.Node;
 
public class Codeforces {
	static int mod= 998244353;
	static char arr[];
	
	public static void main(String[] args) throws Exception {
		PrintWriter out=new PrintWriter(System.out);
		FastScanner fs=new FastScanner();
//		int t=fs.nextInt();
		int t=fs.nextInt();
		outer:while(t-->0) {
			int n=fs.nextInt();
			int p[]=new int[n];
			Arrays.fill(p, -1);
			for(int i=0;i<n;i++) {
				if(p[i]==-1) {
					List<Integer> list=new ArrayList<>();
					Set<Integer> set=new HashSet<>();
					while(true) {
						System.out.println("? "+(i+1));
						int cur=fs.nextInt();
						list.add(cur);
						if(set.contains(cur)) break;
						set.add(cur);
						
					}
					for(int j=0;j<list.size()-1;j++) {
						int cur=list.get(j), next=list.get(j+1);
						p[cur-1]=next;
					}
				}
			}
			StringBuilder ans=new StringBuilder("! ");
			for(int ele:p) {
				ans.append(ele+ " ");
			}
			System.out.println(ans);
		}
		
		
		out.close();
		
	}
	
	
	
	
	
	
	
	
	
	
	// Use this to input code since it is faster than a Scanner
	
 
}
