import java.io.*;
import java.util.*;
public class huge implements Runnable 
{
	private boolean console=false;
	
	ArrayList<Integer> adj[]; int maxi,maxc; int dis[];
	int vis[];
	Queue<Integer> q;
	public void solve() 
	{
		int i;
		int n=in.ni(); maxi=0; maxc=0;
		int a=in.ni()-1,b=in.ni()-1; 
		int da=in.ni(); int db=in.ni();
		adj=new ArrayList[n];
		for(i=0;i<n;i++)
			adj[i]=new ArrayList();
		for(i=0;i<n-1;i++)
		{
			int u=in.ni()-1; int v=in.ni()-1;
			adj[u].add(v); adj[v].add(u); 
		}
		maxc=maxi=0;
		vis=new int[n]; int cur=-1; dfs(1,cur);
		Arrays.fill(vis,0);
		cur=-1; maxc=0; dfs(maxi,cur);
		
		int d=(maxc);
		int ans=1;
		if(db<=2*da||2*da>=d||da>=db)
			ans=0;
		q= new LinkedList();
		vis=new int[n];
		q.add(a);
		dis=new int[n];
		dis[a]=0; int cu=0;
		while(!q.isEmpty()||vis[b]==0)
		{
			int v=q.poll();
			vis[v]=1;
			bfs(v);
		}
		if(dis[b]<=da)
			ans=0;
		if(ans==1)
			out.println("Bob");
		else
			out.println("Alice");
	}
	public void bfs(int v)
	{
		for(int node:adj[v])
		{
			if(vis[node]==0)
			{
				q.add(node);
				dis[node]=dis[v]+1;
			}
		}
	}
	public void dfs(int v,int cur)
	{
		cur++;
		if(cur>maxc)
		{
			maxc=cur; maxi=v;
		}
		vis[v]=1;
		for(int node:adj[v])
		{
			if(vis[node]==0)
				dfs(node,cur);
		}
	}
	@Override
	public void run() {
		try {  init();  } 
		catch (FileNotFoundException e) {  e.printStackTrace();   }

		int t= in.ni();

		while (t-->0) {
			solve();
			out.flush(); }
	}
	private FastInput in;    private PrintWriter out;
	public static void main(String[] args) throws Exception {	 new huge().run();	 }

	private void init() throws FileNotFoundException {
		InputStream inputStream = System.in; 	 OutputStream outputStream = System.out;
		try {  if (!console && System.getProperty("user.name").equals("sachan")) {
			outputStream = new FileOutputStream("/home/sachan/Desktop/output.txt");
			inputStream = new FileInputStream("/home/sachan/Desktop/input.txt");   	}
		}	catch (Exception ignored) {	}
		out = new PrintWriter(outputStream); 	 in = new FastInput(inputStream);
	}
	
}
