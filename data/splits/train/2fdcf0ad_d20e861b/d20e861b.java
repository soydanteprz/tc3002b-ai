import java.util.*;

public class CodeForces{
// 4
//   -3 -5 -2 1
// k  2  4  0 3
static ArrayList<int[]> ans = new ArrayList<>();
    
    public static void add(int a,int b,int c,int d,int e,int f){
        int[] tem=new int[6];
        tem[0]=a;tem[1]=b;tem[2]=c;tem[3]=d;tem[4]=e;tem[5]=f;
        ans.add(tem);
    }
   public static void helper(int[][] arr,int r,int c){
       for(int k=0;k<12;k++){
           if(arr[r][c]==1){
               add(r,c,r+1,c,r,c+1);
               arr[r][c]=1-arr[r][c];
               arr[r+1][c]=1-arr[r+1][c];
               arr[r][c+1]=1-arr[r][c+1];
           }
           if(arr[r][c+1]==1){
                add(r,c+1,r+1,c+1,r,c);
               arr[r][c+1]=1-arr[r][c+1];
               arr[r+1][c+1]=1-arr[r+1][c+1];
               arr[r][c]=1-arr[r][c];
           }
           if(arr[r+1][c]==1){
                add(r+1,c+1,r,c,r+1,c);
               arr[r+1][c+1]=1-arr[r+1][c+1];
               arr[r][c]=1-arr[r][c];
               arr[r+1][c]=1-arr[r+1][c];
           }
           if(arr[r+1][c+1]==1){
                add(r+1,c+1,r+1,c,r,c+1);
               arr[r+1][c+1]=1-arr[r+1][c+1];
               arr[r+1][c]=1-arr[r+1][c];
               arr[r][c+1]=1-arr[r][c+1];
           }
           
       }
       
   } 
   public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		int t=sc.nextInt();
		while(t-->0){
		    int n = sc.nextInt();
		    int m = sc.nextInt();
		    ans.clear();
		    int[][] arr=new int[n][m];
		    for(int i=0;i<n;i++){
		        String str = sc.next();
		        for(int j=0;j<m;j++){
		            arr[i][j]=str.charAt(j)-'0';
		        }
		    }
		    
		    for(int i=0;i<n;i+=2){
		        for(int j=0;j<m;j+=2){
		            if(i==n-1&&j==m-1){
		                helper(arr,i-1,j-1);
		            }else if(i==n-1){
		                helper(arr,i-1,j);
		            }else if(j==m-1){
		                helper(arr,i,j-1);
		            }else{
		                helper(arr,i,j);
		            }
		            
		        }
		    }
		    System.out.println(ans.size());
		    for(int[] move:ans){
		        for(int i=0;i<6;i++){
		            System.out.print(move[i]+1 +" ");
		        }
		        System.out.println();
		    }
		  //  for(int i=0;i<n;i++){
		        
		  //      for(int j=0;j<m;j++){
		  //          System.out.print(arr[i][j] +" ");
		  //      }
		  //      System.out.println();
		  //  }	
		
		    
		}   
		
		
        
	}
}
