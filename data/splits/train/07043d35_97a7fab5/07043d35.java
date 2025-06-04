import java.util.*;
import java.io.*;
 public class C{
     
    static int MOD=1000000007;
    static PrintWriter pw;
    static FastReader sc;
    
       


     
     
    

    
    
    
    // int [] arr=sc.intArr(n);
    static int Max=2*100005;

    // static int [][] arr= new int [Max][5];
    // static int [] tot= new int [Max];
    static ArrayList<ArrayList<Integer>> arr= new ArrayList<>();
    static ArrayList<Integer> tot= new ArrayList<>();
    static int fun(int ch , int n){
    	ArrayList<Integer> list= new ArrayList<>();
    	for(int i=0;i<n;i++){
    		list.add( (arr.get(i).get(ch))-(tot.get(i)-(arr.get(i).get(ch))) );
    	}
    	Collections.sort(list,Collections.reverseOrder());
    	int count=0,sum=0;
    	for(Integer x: list){
    		sum+=x;
    		if(sum>0){
    			count++;
    		}
    	}
    	return count;
    }

    static void solve() throws Exception{
    	int n=sc.nextInt();
    	arr= new ArrayList<>();
    	tot= new ArrayList<>();
    	for(int i=0;i<n;i++){
            arr.add( new ArrayList<>());
    		tot.add(0);
    		for(int j=0;j<5;j++){
    			ArrayList<Integer> list=arr.get(i);
    			list.add(0);
    		}
    	}
    	// pw.println(tot);
    	// pw.println(arr);
    	// return;
    	for(int i=0;i<n;i++){
    		String s=sc.next();
    		tot.set(i,s.length());
    		for(int j=0;j<s.length();j++){
    			ArrayList<Integer> list=arr.get(i);
    			list.set((s.charAt(j)-'a'),list.get((s.charAt(j)-'a'))+1);
    			// list.set((s.charAt(j)-'a'),1);
    		}
    	}
    	int ans =0;
    	for(int i=0;i<5;i++){
    		ans = Math.max(ans,fun(i,n));
    	}
    	pw.println(ans);
 }
    
   public static void main(String[] args) throws Exception{
        try {
            System.setIn(new FileInputStream("input.txt"));
            System.setOut(new PrintStream(new FileOutputStream("output.txt")));
        } catch (Exception e) {
            System.err.println("Error");
        }
        sc= new FastReader();
        pw = new PrintWriter(System.out);

   
        int tc=1;
        tc=sc.nextInt();
        for(int i=1;i<=tc;i++) {
//            pw.printf("Case #%d: "b, i);
            solve();
        }
       
        pw.flush();

    

    }

}
