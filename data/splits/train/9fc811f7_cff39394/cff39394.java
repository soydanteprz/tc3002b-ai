import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.StringTokenizer;

public class D {
	static int diam=0;
	public static int dfs(int a, int p,int[] depth,ArrayList<Integer>[] list) {
		int len=0;
		for(int y:list[a]) {
			if(y==p)
				continue;
			depth[y]=depth[a]+1;
			int cur=1+dfs(y,a,depth,list);
			diam=Math.max(cur+len,diam);
			len=Math.max(cur, len);
		}
		return len;
	}

	public static void main(String[] args) {
		MyScanner s = new MyScanner();
		out = new PrintWriter(new BufferedOutputStream(System.out));
		try {
		int t=s.nextInt();
		while(t-->0) {
		int n=s.nextInt();
		int a=s.nextInt();
		int b=s.nextInt();
		int da=s.nextInt();
		int db=s.nextInt();
		ArrayList<Integer>[] list=new ArrayList[n+1];
		for(int i=0;i<=n;i++)
			list[i]=new ArrayList<Integer>();
		for(int i=0;i<n-1;i++) {
			int u=s.nextInt();
			int v=s.nextInt();
			list[u].add(v);
			list[v].add(u);
		}
		int[] depth=new int[n+1];
		diam=0;
		depth[a]=0;
		int dia=dfs(a,0,depth,list);
		if(2*da>=Math.min(diam, db) || depth[b]<=da) {
			System.out.println("Alice");
		}
		else {
			System.out.println("Bob");
		}
		}
		out.close();
		}catch(Exception e) {
			System.out.println(e);
			return;
		}
	}
	
	public static PrintWriter out;
	
	
	

}
