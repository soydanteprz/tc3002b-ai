import java.util.*;

//CODE FORCES

public class anshulvmc {

    
    public static void sort(int[] a) {
		ArrayList<Integer> l=new ArrayList<>();
		for (int i:a) l.add(i);
		Collections.sort(l);
		for (int i=0; i<a.length; i++) a[i]=l.get(i);
	}
    
    
    
    
    
    
    
    
    
//    public static void gt(int[][] arr,int k) {
//    	int n = arr.length+1;
//    	k = Math.min(k,n+1);
//    	
//    	Node[] nodes = new Node[n];
//    	for(int i=0;i<n;i++) nodes[i] = new Node();
//    	for(int i=0;i<n-1;i++) {
//    		int a = arr[i][0];
//    		int b = arr[i][1];
//    		System.out.println(a+" "+b);
//    		nodes[a].adj.add(nodes[b]);
//    		nodes[b].adj.add(nodes[a]);
//    	}
//    	
//    	ArrayDeque<Node> bfs = new ArrayDeque<>();
//    	for(Node nn:nodes) {
//    		if(nn.adj.size()<2) {
//    			bfs.addLast(nn);
//    			nn.dist=0;
//    		}
//    	}
//    	
//    	while(bfs.size()>0) {
//    		Node nn = bfs.removeFirst();
//    		for(Node a : nn.adj) {
//    			if(a.dist!=-1) continue;
//    			a.usedDegree++;
//    			if(a.adj.size() - a.usedDegree <= 1) {
//    				a.dist = nn.dist+1;
//    				bfs.addLast(a);
//    			}
//    		}
//    	}
//    	
//    	int[] cs = new int[n+1];
//    	for(Node nn:nodes) {
//    		cs[nn.dist]++;
//    	}
//    	for(int i=1;i<cs.length;i++) cs[i]+=cs[i-1];
//    	System.out.println(n-cs[k-1]);
//    }
    
    
    
    
    
    
    
    
	
    
    
    
	public static boolean redblue(int[] num, String chnum) {
    	ArrayList<Integer> blue = new ArrayList<>();
    	ArrayList<Integer> red = new ArrayList<>();
    	for(int i=0;i<chnum.length();i++) {
    		char ch = chnum.charAt(i);
    		if(ch == 'B') {
    			blue.add(num[i]);
    		}
    		else {
    			red.add(num[i]);
    		}
    	}
    	Collections.sort(blue);
    	Collections.sort(red);
//    	System.out.println(blue);
//    	System.out.println(red);
    	for(int i=0;i<blue.size();i++) {
    		if(blue.get(i) >= i+1) {
    			
    		}
    		else {
    			return false;
    		}
    	}
    	for(int i=0;i<red.size();i++) {
    		if(red.get(i) > i+1 + blue.size()) {
//    			System.out.println(red.get(i)+" "+(i+1 + blue.size()));
    			return false;
    		}
    	}
    	return true;
    }
    
    public static void main(String args[])
    
    
    {	
    	Scanner scn = new Scanner(System.in);
    	int test = scn.nextInt();
    	for(int i=0;i<test;i++) {
    		int size = scn.nextInt();
    		int[] arr = new int[size];
    		for(int j=0;j<size;j++) {
    			arr[j] = scn.nextInt();
    		}
    		String str = scn.next();
    		boolean f = redblue(arr,str);
    		if(f) {
    			System.out.println("YES");
    		}
    		else {
    			System.out.println("NO");
    		}
    	}
//    	int n = 1;
//    	int[] dp = new int[2];
//    	System.out.println(fact(n,dp));
    			
    }

}
