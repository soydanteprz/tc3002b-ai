import java.io.*;
import java.util.*;



public class A {

	

	


	static Scanner sc = new Scanner(System.in);
	static StringBuilder out = new StringBuilder();

	public static void main(String[] args) throws IOException {

		int t = 1;// sc.nextInt();

		while (t-- > 0) {

			A run = new A();

			run.run();

		}

		System.out.println(out);

	}

	

	public void run() throws IOException {

//	int k = sc.nextInt();

//		if(100%k==0)out.append((100/k)+"\n");
//		
//		else {
//			out.append(100+"\n");
//		}

//		int ans=(int)1e9;
//		for(int i=1;i<=k;i++) {
//			
//			
//			if(i*100%k==0) {
//				
//				ans=Math.min(ans,i*)
//				
//				
//			}
//			
//		}

//		
//		int ans = 100 - k;
//		if (k == 100)
//			out.append(1);
//		else {
//			int g = gcd(k, ans);
//			out.append((k / g) + (ans / g));
//		}
//		out.append("\n");
//		

//		int n = sc.nextInt();
//		int a[] = new int[n];
//		
//		for(int i=0;i<n;i++)a[i]=sc.nextInt();
//		int min = Integer.MAX_VALUE, mx = Integer.MIN_VALUE;
//		for (int i = 0; i < n; i++) {
//			min = Math.min(min, a[i]);
//			mx = Math.max(mx, a[i]);
//		}
//		int b[] = a.clone();
//		sort(b,n);
//		if (Arrays.equals(a, b)) {
//			out.append(0);
//		} else if (a[0] == min || a[n - 1] == mx) {
//			out.append(1);
//		} else if (a[0] == mx && a[n - 1] == min) {
//			out.append(3);
//		} else {
//			out.append(2);
//		}
//		out.append("\n");

		int n = sc.nextInt();

		int a[] = new int[n];

		TreeSet<Integer> ts = new TreeSet<>();

		ArrayList<Integer> a1 = new ArrayList<>();
		ArrayList<Integer> b = new ArrayList<>();
		for (int i = 0; i < n; i++) {

			a[i] = sc.nextInt();
			if (a[i] == 1)
				a1.add(i);
			else
				b.add(i);

		}

		int dp[] = new int[n];
		int sum = 0;

		memo = new int[n][n];

		for (int tem[] : memo)
			Arrays.fill(tem, -1);

		sum = solve(0, 0, a1, b);
		out.append(sum + "\n");
	}

	static int memo[][];

	static int solve(int i, int j, ArrayList<Integer> a, ArrayList<Integer> b) {

		if(i==a.size())return 0;
		if(a.size()-i>b.size()-j)return (int)1e9;
		if(j==b.size())return (int)1e9;
		if(memo[i][j]!=-1)return memo[i][j];
		int ans = (int) 1e9;

		
		
		

			ans = Math.min(solve(i + 1, j+1 , a, b) + Math.abs(a.get(i) - b.get(j)),solve(i,j+1,a,b));
		
		
		return memo[i][j]=ans;

	}

	

	

	

//	static Reader sc = new Reader();

	

}
