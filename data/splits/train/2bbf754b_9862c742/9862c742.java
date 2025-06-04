import java.util.*;
public class Main
{
	public static void main(String[] args) {
		Scanner s  = new Scanner(System.in);
		int t = s.nextInt();
		while(t-- > 0){
		    int n = s.nextInt();
		    int[] arr = new int[n];
		    for(int i=0;i<n;i++)
		        arr[i] = s.nextInt();
		    String str = s.next();
		    List<Integer> left = new ArrayList<>();
		    List<Integer> right = new ArrayList<>();
		    for(int i=0;i<arr.length;i++){
		        if(str.charAt(i) == 'B')
		            left.add(arr[i]);
		        else right.add(arr[i]);
		    }
		    Collections.sort(left);
		    Collections.sort(right);
		    boolean flag = false;
		    for(int i=0;i<left.size();i++)
		        if(left.get(i) < i+1){
		            flag = true;
		            break;
		        }
		    if(!flag){
		        for(int i=0;i<right.size();i++)
    		        if(right.get(i) > left.size()+i+1){
    		            flag = true;
    		            break;
    		        }
		    }
		    if(flag)
		        System.out.println("NO");
		    else System.out.println("YES");
		}
	}
}
