import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.abs;
import static java.lang.System.out;
import java.util.*;
import java.io.*;
import java.math.*;

public class Template {

	static int mod = 1000000007;

	public static void main(String[] args) {
		FastScanner sc = new FastScanner();
		int yo = sc.nextInt();
		while (yo-- > 0) {
			int n = sc.nextInt();
            int a = sc.nextInt()-1;
            int b = sc.nextInt()-1;
            int da = sc.nextInt();
            int db = sc.nextInt();
            
            List<List<Integer>> list = new ArrayList<>();
            for(int i = 0; i < n; i++) list.add(new ArrayList<>());
            
            for(int i = 0; i < n-1; i++){
                int x = sc.nextInt()-1;
                int y = sc.nextInt()-1;
                list.get(x).add(y);
                list.get(y).add(x);
            }
            
            for(int i = 0; i <= n; i++) depth[i] = 0;
            diam = 0;
            dfs(a,-1,list);
            
            if(2 * da >= min(diam, db) || depth[b] <= da){
                out.println("Alice");
            }
            else {
                out.println("Bob");
            }
            
		}
	}
    
    static int[] depth = new int[200001];
    static int diam = 0;
    static int dfs(int x, int p, List<List<Integer>> list) {
        int len = 0;
        List<Integer> ne = list.get(x);
        for(int y : ne) {
            if(y != p) {
                depth[y] = depth[x] + 1;
                int cur = 1 + dfs(y, x,list);
                diam = max(diam, cur + len);
                len = max(len, cur);
            }
        }
        return len;
    }
   

	

	

	

	

	

	

	

	//	For Input.txt and Output.txt	
	//	FileInputStream in = new FileInputStream("input.txt");
	//	FileOutputStream out = new FileOutputStream("output.txt");
	//	PrintWriter pw = new PrintWriter(out);
	//	Scanner sc = new Scanner(in);
}
