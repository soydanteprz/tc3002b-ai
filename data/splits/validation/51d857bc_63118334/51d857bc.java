import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Round659 {

	static int rec = 0;
	static int X[] = { -1, 0, 0, 1 };
	static int Y[] = { 0, -1, 1, 0 };
	static long mod = 1000000007;
	static int last=0;
	static int maxDepth=0;
	

	

	

	

	

	public static List<Integer>[] initAdjacency(int n, int e, Reader scan, boolean type) throws IOException {

		List<Integer> adj[] = new ArrayList[n + 1];

		for (int i = 0; i < e; i++) {
			int u = scan.nextInt();
			int v = scan.nextInt();

			if (adj[u] == null)
				adj[u] = new ArrayList<>();
			if (type && adj[v] == null)
				adj[v] = new ArrayList<>();

			adj[u].add(v);

			if (type)
				adj[v].add(u);

		}

		return adj;
	}

	public static void main(String[] args) throws IOException {

		Reader scan = new Reader();
//		Scanner scan = new Scanner(System.in);

		int t = scan.nextInt();
		while (t-- > 0) {
			
			D(scan);
		}

	}
	
	
	public static void D(Reader scan) throws IOException {
		
		int n=scan.nextInt();
		int a=scan.nextInt();
		int b=scan.nextInt();
		int da=scan.nextInt();
		int db=scan.nextInt();
		
		List<Integer> adj[]=initAdjacency(n, n-1, scan, true);
		
		int d= dist(a, 0, adj, b);
		
		if(d<=da) {
			System.out.println("Alice");
			return;
		}
		
		if(db<=2*da) {
			System.out.println("Alice");
			return;
		}
		
		MyPair far1= farthest(1, 0, adj);
		MyPair far2=farthest(far1.weight, 0, adj);
		
		int diameter= far2.value-1;
		
		
		if(diameter<=2*da) {
			System.out.println("Alice");
		}else {
			System.out.println("Bob");
		}
		
		
//		int val=d-2*a;
//		
//		if(val>0) {
//			System.out.println("Bob");
//			return;
//		}
//		
//		int dep=0;
//		for(Integer x: adj[a]) {
//			if(x==last) continue;
//			
//			dep=Math.max(getDepth(x, a, adj), dep);
//		}
//		
//		if(dep>=val) {
//			System.out.println("Bob");
//		}else {
//			System.out.println("Alice");
//		}
		
		
		
	}
	
	public static MyPair farthest(int i, int parent, List<Integer> adj[]) {
		
		int dist=0;
		int node= i;
		
		for(Integer x: adj[i]) {
			if(x==parent) continue;
			
			MyPair recAns= farthest(x, i, adj);
			if(recAns.value>dist) {
				dist= recAns.value;
				node= recAns.weight;
			}
		}
		
		return new MyPair(dist+1, node);
		
	}
	
	
	
	public static int dist(int i, int parent, List<Integer> adj[], int target) {
		
		if(i==target) return 0;
		
		for(Integer x: adj[i]) {
			if(x==parent) continue;
			
			int recAns= dist(x, i, adj, target);
			if(recAns!=-1) {
				last=x;
				return recAns+1;
			}
		}
		
		return -1;
		
		
	}
	
	
	
	
	
	
	

}

class MyPair {

	int value;
	int weight;

	public MyPair(int value, int w) {
		this.value = value;
		weight = w;
	}

}
