import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

public class PC3C {
	static PrintWriter out = new PrintWriter(System.out);
	static MyFastReaderPC3C in = new MyFastReaderPC3C();
	static long mod = (long) (1e9 + 7);

	public static void main(String[] args) throws Exception {
		
		int test = i();
		while (test-- > 0) {
			 
             int n=i();
             int[] arr=arrI(n);
             String s=string();
             
             ArrayList<Integer> lR=new ArrayList<>();

             ArrayList<Integer> lB=new ArrayList<>();
             for(int i=0;i<n;i++) {
            	
            	if(s.charAt(i)=='R') lR.add(arr[i]);
            	else lB.add(arr[i]);
            	 
             }
             
             
             Collections.sort(lB);
             Collections.sort(lR,Collections.reverseOrder());
             
             
             int k=1;
             boolean st=true;
             for(int i=0;i<lB.size();i++) {
            	 
            	 if(lB.get(i)>=k) {
            		k+=1; 
            	 }
            	 else {
            		st=false;
            		break;
            	 }
            		 
             }
             boolean st2=true;
               k=n;
               for(int i=0;i<lR.size();i++) {
            	 
            	 if(lR.get(i)>k) {
            	   st2=false;
            	   break;
            	 }
            	 else {
            		k-=1; 
            	 }
            		 
             }
               
               if(st && st2) out.print("YES");
               else out.print("NO");
             
             
			out.print("\n");
			out.flush();
		}
		out.close();
	}

	

	static void sort(long[] a) // check for long
	{
		ArrayList<Long> l = new ArrayList<>();
		for (long i : a)
			l.add(i);
		Collections.sort(l);
		for (int i = 0; i < a.length; i++)
			a[i] = l.get(i);
	}

	static void sort(int[] a) {
		ArrayList<Integer> l = new ArrayList<>();
		for (int i : a)
			l.add(i);
		Collections.sort(l);
		for (int i = 0; i < a.length; i++)
			a[i] = l.get(i);
	}

	

	

	

	

	

	

	

	

	

	

	// Debugging Functions Starts

	static void print(char A[]) {
		for (char c : A)
			System.out.print(c + " ");
		System.out.println();
	}

	static void print(boolean A[]) {
		for (boolean c : A)
			System.out.print(c + " ");
		System.out.println();
	}

	static void print(int A[]) {
		for (int a : A)
			System.out.print(a + " ");
		System.out.println();
	}

	static void print(long A[]) {
		for (long i : A)
			System.out.print(i + " ");
		System.out.println();

	}

	static void print(ArrayList<Integer> A) {
		for (int a : A)
			System.out.print(a + " ");
		System.out.println();
	}

	// Debugging Functions END
	// ----------------------
	// IO FUNCTIONS STARTS
	

	public static Map<Character, Integer> mapSortByValue(Map<Character, Integer> hm) {
		// Create a list from elements of HashMap
		List<Map.Entry<Character, Integer>> list = new LinkedList<Map.Entry<Character, Integer>>(hm.entrySet());

		// Sort the list
		Collections.sort(list, new Comparator<Map.Entry<Character, Integer>>() {
			
		});

		// put data from sorted list to hashmap
		Map<Character, Integer> temp = new LinkedHashMap<Character, Integer>();
		for (Map.Entry<Character, Integer> aa : list) {
			temp.put(aa.getKey(), aa.getValue());
		}
		return temp;
	}
	static String string() {
		return in.nextLine();
	}

	static int i() {
		return in.nextInt();
	}

	static long l() {
		return in.nextLong();
	}

	static int[] arrI(int N) {
		int A[] = new int[N];
		for (int i = 0; i < N; i++) {
			A[i] = in.nextInt();
		}
		return A;
	}

	

}


